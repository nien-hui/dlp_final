import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv
from torch_pruning.pruner.function import BasePruningFunc        # 型別提示用
import torch
import torch_pruning as tp
from torch_geometric.nn.dense import Linear          # PyG 專用 Linear
from typing import Sequence
from torch_geometric.nn.dense.linear import is_uninitialized_parameter

# ------------------------------------------------------------------
# 1) 專門裁剪 torch_geometric.nn.dense.Linear
# ------------------------------------------------------------------
class PyGLinearPruner(BasePruningFunc):
    """Pruner for torch_geometric.nn.dense.Linear."""
    TARGET_MODULES = (Linear,)

    # ---------- 裁剪「輸出」通道 ----------
    def prune_out_channels(self, layer: Linear, idxs: Sequence[int]) -> Linear:
        # 若還沒 lazy init，就先 materialize（避免不能切 slice）
        if is_uninitialized_parameter(layer.weight):
            raise RuntimeError("Linear weight still un-initialized; "
                               "請先跑一次 forward 或手動 initialize_parameters")

        keep = sorted(set(range(layer.out_channels)) - set(idxs))

        # 0 維 (rows) = out_channels
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep, 0)

        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep, 0)

        layer.out_channels = len(keep)
        return layer

    # ---------- 裁剪「輸入」通道 ----------
    def prune_in_channels(self, layer: Linear, idxs: Sequence[int]) -> Linear:
        if is_uninitialized_parameter(layer.weight):
            raise RuntimeError("Linear weight still un-initialized; "
                               "請先跑一次 forward 或手動 initialize_parameters")

        keep = sorted(set(range(layer.in_channels)) - set(idxs))

        # 1 維 (cols) = in_channels
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep, 1)

        layer.in_channels = len(keep)
        return layer

    # ---------- 報告可剪通道數 ----------
    def get_out_channels(self, layer): return layer.out_channels
    def get_in_channels (self, layer): return layer.in_channels


class GCNConvPruner(BasePruningFunc):
    """Pruner for torch_geometric.nn.GCNConv."""
    TARGET_MODULES = (GCNConv,)

    # 把 Linear 裁剪邏輯委派出去
    _linear_pruner = PyGLinearPruner()

    # ---------- 裁剪「輸出」通道 ----------
    def prune_out_channels(self, layer: GCNConv, idxs: Sequence[int]) -> GCNConv:
        keep = sorted(set(range(layer.out_channels)) - set(idxs))

        # 1) 裁內部 Linear
        self._linear_pruner.prune_out_channels(layer.lin, idxs)

        # 2) 同步外層屬性 / bias
        layer.out_channels = len(keep)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep, 0)

        # 3) 清快取
        layer._cached_edge_index = None
        layer._cached_adj_t      = None
        return layer

    # ---------- 裁剪「輸入」通道 ----------
    def prune_in_channels(self, layer: GCNConv, idxs: Sequence[int]) -> GCNConv:
        keep = sorted(set(range(layer.in_channels)) - set(idxs))

        self._linear_pruner.prune_in_channels(layer.lin, idxs)
        layer.in_channels = len(keep)
        return layer

    # ---------- 報告可剪通道數 ----------
    def get_out_channels(self, layer): return layer.out_channels
    def get_in_channels (self, layer): return layer.in_channels


class GINConvPruner(BasePruningFunc):
    """Pruner for torch_geometric.nn.GCNConv."""
    TARGET_MODULES = (GINConv,)

    # 把 Linear 裁剪邏輯委派出去
    _linear_pruner = PyGLinearPruner()

    # ---------- 裁剪「輸出」通道 ----------
    def prune_out_channels(self, layer: GINConv, idxs: Sequence[int]) -> GINConv:
        keep = sorted(set(range(layer.out_channels)) - set(idxs))

        # 1) 裁內部 Linear
        self._linear_pruner.prune_out_channels(layer.lin, idxs)

        # 2) 同步外層屬性 / bias
        layer.out_channels = len(keep)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep, 0)

        # 3) 清快取
        layer._cached_edge_index = None
        layer._cached_adj_t      = None
        return layer

    # ---------- 裁剪「輸入」通道 ----------
    def prune_in_channels(self, layer: GINConv, idxs: Sequence[int]) -> GINConv:
        keep = sorted(set(range(layer.in_channels)) - set(idxs))

        self._linear_pruner.prune_in_channels(layer.lin, idxs)
        layer.in_channels = len(keep)
        return layer

    # ---------- 報告可剪通道數 ----------
    def get_out_channels(self, layer): return layer.out_channels
    def get_in_channels (self, layer): return layer.in_channels

