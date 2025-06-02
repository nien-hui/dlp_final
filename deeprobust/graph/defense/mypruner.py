import torch
import torch.nn as nn
import typing 
from torch_pruning.pruner.algorithms import BasePruner
from torch_pruning.pruner.algorithms.scheduler import linear_scheduler
from torch_pruning.pruner import function
from torch_pruning import ops

from .pruner_function import GCNConvPruner

from torch_pruning.pruner.function import BasePruningFunc        # 型別提示用
import torch_pruning as tp
from torch_pruning.pruner.function import BasePruningFunc
from torch_geometric.nn.dense import Linear          # PyG 專用 Linear
from torch_geometric.nn import GCNConv
from typing import Sequence
from torch_geometric.nn.dense.linear import is_uninitialized_parameter


class MyGroupNormPruner(BasePruner):
    """
    A regularization-based pruner for group-level pruning. 
    This pruner implement an regularization method that spasify parameters by group.
    https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html

    Args:
        model (nn.Module): A to-be-pruned model
        example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
        importance (Callable): importance estimator. 
        reg (float): regularization coefficient. Default: 1e-5.
        alpha (float): regularization scaling factor, [2^0, 2^alpha]. Default: 4.
        global_pruning (bool): enable global pruning. Default: False.
        pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
        pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio. Will cover pruning_ratio if specified. Default: None.
        max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
        iterative_steps (int): number of steps for iterative pruning. Default: 1.
        iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
        ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
        round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
        isomorphic (bool): enable isomorphic pruning. Default: False. https://arxiv.org/abs/2407.04616
        in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
        out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
        num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
        prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
        prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
        head_pruning_ratio (float): head pruning ratio. Default: 0.0.
        head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
        customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
        unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
        root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
        forward_fn (Callable): A function to execute model.forward. Default: None.
        output_transform (Callable): A function to transform network outputs. Default: None.
        channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
        ch_sparsity (float): the same as pruning_ratio. Default: None.
        ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.

    """
    def __init__(
        self,
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-4, # regularization coefficient
        alpha=4, # regularization scaling factor, [2^0, 2^alpha]
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratio, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to
        isomorphic: bool = False, # enable isomorphic pruning (ECCV 2024, https://arxiv.org/abs/2407.04616) if global_pruning=True. 

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
    ):
        super(MyGroupNormPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            isomorphic=isomorphic,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict
        )

        self.reg = reg
        self.alpha = alpha
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        self.cnt = 0

    def update_regularizer(self):
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

    @torch.no_grad()
    def regularize(self, model, alpha=2**4, bias=False):
        for i, group in enumerate(self._groups):
            ch_groups = self._get_channel_groups(group)
            imp = self.estimate_importance(group).sqrt()
            if torch.any(torch.isnan(imp)):
                continue
            gamma = alpha ** ((imp.max() - imp) / (imp.max() - imp.min()))

            # --------------------------------------------------------------
            #  每一個 dependency edge 逐一正則化
            # --------------------------------------------------------------
            for i, (dep, idxs) in enumerate(group): 

                layer     = dep.target.module
                prune_fn  = dep.handler
                root_idxs = group[i].root_idxs

                # ────────── 1. Conv / Linear / GCNConv  (★ out channels) ──────────
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,     
                ]:  
                    
                    if layer.weight.grad is None:
                        continue


                    _gamma = torch.index_select(
                        gamma, 0, torch.tensor(root_idxs, device=gamma.device)
                    )

                    print(idxs)

                    w = layer.weight.data[idxs]
                    g = w * _gamma.view(-1, *([1] * (w.dim() - 1)))
                    layer.weight.grad.data[idxs] += self.reg * g

                    if bias and getattr(layer, "bias", None) is not None:
                        b = layer.bias.data[idxs]
                        g = b * _gamma
                        layer.bias.grad.data[idxs] += self.reg * g



                # ────────── 2. Conv / Linear / GCNConv  (★ in channels) ──────────
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    if layer.weight.grad is None:
                        continue

                    # GCNConv 其權重仍是 Linear，維度 = [out, in]
                    if prune_fn == function.prune_conv_in_channels and layer.groups > 1:
                        gamma_ = gamma[: len(idxs) // ch_groups]
                        idxs_  = idxs[: len(idxs) // ch_groups]
                    else:
                        gamma_, idxs_ = gamma, idxs

                    _gamma = torch.index_select(
                        gamma_, 0, torch.tensor(root_idxs, device=gamma.device) 
                    )

                    w = layer.weight.data[:, idxs_]
                    g = w * _gamma.view(1, -1, *([1] * (w.dim() - 2)))
                    layer.weight.grad.data[:, idxs_] += self.reg * g

                # ────────── 3. BatchNorm ──────────
                elif prune_fn == function.prune_batchnorm_out_channels:
                    if layer.affine is None or layer.weight.grad is None:
                        continue

                    _gamma = torch.index_select(
                        gamma, 0, torch.tensor(root_idxs, device=gamma.device)
                    )

                    w = layer.weight.data[idxs]
                    g = w * _gamma
                    layer.weight.grad.data[idxs] += self.reg * g

                    if bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        g = b * _gamma
                        layer.bias.grad.data[idxs] += self.reg * g

                elif prune_fn == self.DG.CUSTOMIZED_PRUNERS[GCNConv].prune_out_channels:
                    _gamma = torch.index_select(
                        gamma, 0, torch.tensor(root_idxs, device=gamma.device)
                    )

                    # layer.lin.weight shape = [out, in]
                    w = layer.lin.weight.data[idxs]                       # [prune, in]
                    g = w * _gamma.view(-1, 1)                            # broadcast
                    layer.lin.weight.grad.data[idxs] += self.reg * g

                    if bias and layer.bias is not None:
                        b = layer.bias.data[idxs]                         # [prune]
                        g = b * _gamma
                        layer.bias.grad.data[idxs] += self.reg * g 


                elif prune_fn == self.DG.CUSTOMIZED_PRUNERS[GCNConv].prune_in_channels:
                    if layer.lin.weight.grad is None:
                        return

                    _gamma = torch.index_select(
                        gamma, 0, torch.tensor(root_idxs, device=gamma.device) 
                    ) 

                    # layer.lin.weight shape = [out, in]
                    w = layer.lin.weight.data[:, idxs]                    # [out, prune]
                    g = w * _gamma.view(1, -1)                            # broadcast 
                    layer.lin.weight.grad.data[:, idxs] += self.reg * g 
                    

        self.cnt += 1


