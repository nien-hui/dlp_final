from .mypruner import MyGroupNormPruner
from .pruner_function import PyGLinearPruner, GCNConvPruner, GINConvPruner
import torch
from torch_geometric.nn.dense import Linear
from torch_geometric.nn import GCNConv, GINConv
import torch_pruning as tp
from torch_pruning.pruner.importance import GroupMagnitudeImportance

def get_model_parameters_number(model): 
    params_num = sum(p.numel() for p in model.parameters()) 
    return params_num 

class GCNMagnitudeImportance(GroupMagnitudeImportance): 
    """讓 GCNConv 也能用 magnitude 當作重要度"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 把 GCNConv 加進目標清單（父類別會用來過濾 layer）
        self.target_types = tuple(self.target_types) + (GCNConv,) 
        ### 原target_types: list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]) 

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        for i, (dep, idxs) in enumerate(group): 
            layer     = dep.layer                          # 真正的 nn.Module 
            prune_fn  = dep.pruning_fn                     # 觸發的剪枝函式 
            root_idxs = group[i].root_idxs                 # 在 root layer 的索引 

            # ---------- 專門處理 GCNConv ----------
            if isinstance(layer, GCNConv):
                # 「輸出」通道：權重在 layer.lin.weight 的「列」
                if prune_fn.__name__ == "prune_out_channels":
                    W = layer.lin.weight.data[idxs].flatten(1)
                    local_imp = W.abs().pow(self.p).sum(1)

                # 「輸入」通道：權重要先轉置看「行」
                elif prune_fn.__name__ == "prune_in_channels":
                    W = layer.lin.weight.data.transpose(0,1).flatten(1)[idxs]
                    local_imp = W.abs().pow(self.p).sum(1)

                else:            # 不是 in/out channel，直接跳過
                    continue

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                # bias（選擇性）
                if self.bias and layer.bias is not None and \
                   prune_fn.__name__ == "prune_out_channels":
                    b_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(b_imp)
                    group_idxs.append(root_idxs)

                continue   # GCNConv 已處理完，進下一 dep

            # ---------- 其他層 → 交給父類別 ----------
            # 讓父類別跑一次，回傳可能是 None／Tensor
            parent_imp = super().__call__(group[i:i+1]) 
            if parent_imp is not None:
                group_imp.append(parent_imp) 
                group_idxs.append(root_idxs) 

        if len(group_imp) == 0:        # 這個 group 完全沒有可計算的層
            return None 

        # 套用父類別的 reduce / normalize 流程
        reduced = self._reduce(group_imp, group_idxs)
        reduced = self._normalize(reduced, self.normalizer) 
        return reduced 

class Pruner_agent(): 

    def __init__(self, model, example_inputs):

        def forward_fn(model, inputs): 
            return model(*inputs)
        importance = GCNMagnitudeImportance(p=2, group_reduction="mean", normalizer="mean")
        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000: 
                ignored_layers.append(m) # DO NOT prune the final classifier! 
        
        ignored_layers.append(model.gc2)

        pruner = MyGroupNormPruner( # We can always choose BasePruner if sparse training is not required.
            model,
            example_inputs, 
            importance=importance, 
            pruning_ratio=0.2, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256} 
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks 
            ignored_layers=ignored_layers, 
            round_to=1, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html 
            forward_fn = forward_fn,
            customized_pruners = {
                Linear : PyGLinearPruner(),   # PyG Linear 
                GCNConv: GCNConvPruner(),     # 外層 GCNConv 
                GINConv: GINConvPruner(),     # 外層 GCNConv 
            },
            root_module_types  = (tp.ops.TORCH_CONV, tp.ops.TORCH_LINEAR, GCNConv, GINConv), 
            reg = 1e-3
        ) 

        self.pruner = pruner
        self.model = model
        

    def pruning(self):
        base_nparams = get_model_parameters_number(self.model) 
        # tp.utils.print_tool.before_pruning(model) # or print(model) 
        self.pruner.step() 
        # tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning 
        nparams = get_model_parameters_number(self.model) 
        # print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M") 
        # print(f"#Params: {base_nparams/1e6} M -> {nparams/1e6} M") 

    def regularizing(self):
        self.pruner.regularize(self.model)

    def update_regularize(self):
        self.pruner.update_regularizer()
        