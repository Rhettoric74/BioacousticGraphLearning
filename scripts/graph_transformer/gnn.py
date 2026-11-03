from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _GCNStyleSAGEConv(nn.Module):
    """GraphSAGE's GCN-style variant: transform mean(self + neighbors).
    For aggregate="gcn".
    This differs from a standard GCNConv's symmetric degree normalization.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv = SAGEConv(hidden_dim, hidden_dim, aggr="mean", root_weight=False)

    def forward(self, x, edge_index):
        from torch_geometric.utils import add_self_loops, remove_self_loops
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        return self.conv(x, edge_index)


class GNNModel(nn.Module):
    """Node-level GraphSAGE/GATv2 with species and optional auxiliary heads.

    ``x`` is [N, feature_dim]; edges use PyG's source -> destination convention.
    By default all input columns are audio. For concatenated audio + location
    features, set ``audio_dim`` to the number of leading audio columns; remaining
    columns are location features. This allows separate modality masking and
    keeps audio reconstruction at the Perch embedding dimension.

    Output keys match the Transformer, but use a node axis rather than walk and
    modality axes: species [N, C], location_vector [N, 3], coordinates [N, 2],
    hidden [N, H], and audio [N, audio_dim] when audio_proj=True. Species labels
    are never model inputs. The original constructor arguments are preserved;
    num_layers and audio_dim are optional additions.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int,
                 gnn_type: str = "sage", num_heads: int = 4, dropout: float = 0.1,
                 aggregate: str = "mean", audio_proj: bool = False,
                 num_layers: int = 3, audio_dim: int | None = None):
        super().__init__()
        if gnn_type not in ("sage", "gat"):
            raise ValueError(f"gnn_type must be 'sage' or 'gat', got {gnn_type}")
        if min(feature_dim, hidden_dim, num_classes, num_layers) < 1:
            raise ValueError("Feature dimensions, class count and num_layers must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        audio_dim = feature_dim if audio_dim is None else audio_dim
        if not 1 <= audio_dim <= feature_dim:
            raise ValueError("audio_dim must be between 1 and feature_dim")
        if gnn_type == "gat" and (num_heads < 1 or hidden_dim % num_heads != 0):
            raise ValueError("For GATv2, num_heads must be positive and divide hidden_dim")
        if gnn_type == "sage" and aggregate not in ("mean", "lstm", "gcn", "pool", "max", "sum"):
            raise ValueError("Unsupported GraphSAGE aggregate: " + aggregate)

        self.gnn_type, self.aggregate = gnn_type, aggregate
        self.audio_proj = audio_proj
        self.feature_dim, self.audio_dim = feature_dim, audio_dim
        self.num_layers = num_layers
        self.config = dict(feature_dim=feature_dim, hidden_dim=hidden_dim,
                           num_classes=num_classes, gnn_type=gnn_type,
                           num_heads=num_heads, dropout=dropout, aggregate=aggregate,
                           audio_proj=audio_proj, num_layers=num_layers, audio_dim=audio_dim)
        self.input_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        if gnn_type == "sage":
            from torch_geometric.nn import SAGEConv
            for _ in range(num_layers):
                if aggregate == "gcn":
                    layer = _GCNStyleSAGEConv(hidden_dim)
                else:
                    layer = SAGEConv(hidden_dim, hidden_dim,
                                     aggr="max" if aggregate == "pool" else aggregate,
                                     project=(aggregate == "pool"))
                self.gnn_layers.append(layer)
        else:
            from torch_geometric.nn import GATv2Conv
            for _ in range(num_layers):
                self.gnn_layers.append(GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, concat=True))

        self.species_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
        self.audio_head = nn.Linear(hidden_dim, audio_dim) if audio_proj else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                mask_audio: bool = False, mask_location: bool = False,
                location_mask_prob: float = 0.0):
        """Mask raw modalities before projection and message passing.

        Masks apply to every supplied node, including context nodes. Masking
        location features does not remove geographic information in the edges.
        """
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"x must have shape [N, {self.feature_dim}], got {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer node indices")
        if not 0 <= location_mask_prob <= 1:
            raise ValueError("location_mask_prob must be in [0, 1]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        if mask_audio or mask_location or (self.training and location_mask_prob):
            x = x.clone()
            if mask_audio:
                x[:, :self.audio_dim] = 0
            if mask_location:
                x[:, self.audio_dim:] = 0
            elif self.training and location_mask_prob and self.audio_dim < self.feature_dim:
                keep = torch.rand((len(x), 1), device=x.device) >= location_mask_prob
                x[:, self.audio_dim:] = x[:, self.audio_dim:] * keep

        if self.gnn_type == "sage":
            from torch_geometric.utils import remove_self_loops
            # SAGE has a separate root term; the gcn variant adds self once.
            edge_index, _ = remove_self_loops(edge_index)
            if self.aggregate == "lstm":
                edge_index = edge_index[:, edge_index[1].argsort(stable=True)]

        h = self.input_proj(x)
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            h = norm(layer(h, edge_index))
            if i < len(self.gnn_layers) - 1:
                h = self.dropout(F.gelu(h))

        location_vector = F.normalize(self.location_head(h), dim=-1, eps=1e-8)
        latitude = torch.rad2deg(torch.asin(location_vector[..., 2].clamp(-1.0, 1.0)))
        longitude = torch.rad2deg(torch.atan2(location_vector[..., 1], location_vector[..., 0]))
        out = {"species": self.species_head(h), "location_vector": location_vector,
               "coordinates": torch.stack((latitude, longitude), dim=-1), "hidden": h}
        if self.audio_head is not None:
            out["audio"] = self.audio_head(h)
        return out
