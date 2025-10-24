import torch.nn as nn
import torch
from torch_geometric.nn import SetTransformerAggregation, MinAggregation
from .layers import *

class StructureModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.Tanh()):
        super(StructureModel, self).__init__()
        
        self.activation = activation
        self.structural_refinement = StructuralFeatureRefiner(768, hidden_channels, out_channels, dropout, activation)
        
        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_struct = self.structural_refinement(x_struct, edge_index)
        
        x_struct = self.aggr(x_struct[edge_index[0]], edge_index[1])

        x_struct = self.classifier(x_struct)

        return x_struct

class SemanticModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.Tanh()):
        super(SemanticModel, self).__init__()
        
        self.activation = activation
        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        
        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x = self.semantic_refinement(x, edge_index)
        
        x = self.aggr(x[edge_index[0]], edge_index[1])

        x = self.classifier(x)
        
        return x

class NodeSemanticAndStructureModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.LeakyReLU()):
        super(NodeSemanticAndStructureModel, self).__init__()
        
        self.activation = activation
        self.structural_refinement = StructuralFeatureRefiner(768, hidden_channels, out_channels, dropout, activation)
        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        self.node_fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_channels),
        )
        
        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_struct = self.structural_refinement(x_struct, edge_index)
        x = self.semantic_refinement(x, edge_index)
        
        x = torch.cat([x, x_struct], dim=1)
        x = self.node_fusion(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])

        x = self.classifier(x)
        return x

class NodeAndHyperedges(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.Tanh()):
        super(NodeAndHyperedges, self).__init__()
        
        self.activation = activation
        self.semantic_hyperedge_refinement = SemanticHyperedgeRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)

        self.edge_fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_channels),
        )
        
        self.attention_fusion = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.attention_fusion2 = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.attention_fusion3 = NodeHyperedgeAttention(hidden_channels, hidden_channels)

        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_e = self.semantic_hyperedge_refinement(x_e)
        
        x = self.semantic_refinement(x, edge_index)

        x, x_e = self.attention_fusion(x, x_e, edge_index)
        x, x_e = self.attention_fusion2(x, x_e, edge_index)
        x, x_e = self.attention_fusion3(x, x_e, edge_index)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = torch.cat([x, x_e], dim=1)
        x = self.edge_fusion(x)

        x = self.classifier(x)
        return x
    
class FullModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        activation: nn.Module = nn.Tanh()
    ):
        super(FullModel, self).__init__()

        self.activation = activation

        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        self.structural_refinement = StructuralFeatureRefiner(768, hidden_channels, out_channels, dropout, activation)
        self.semantic_hyperedge_refinement = SemanticHyperedgeRefiner(in_channels, hidden_channels, out_channels, dropout, activation)

        self.node_fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_channels),
        )

        self.attention_fusion1 = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.attention_fusion2 = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.attention_fusion3 = NodeHyperedgeAttention(hidden_channels, hidden_channels)

        self.edge_fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_channels),
        )

        self.aggr = MinAggregation()

        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)

    def forward(self, x, x_struct, x_e, edge_index):
        
        x = self.semantic_refinement(x, edge_index)
        x_struct = self.structural_refinement(x_struct, edge_index)
        x_e = self.semantic_hyperedge_refinement(x_e)

        x = torch.cat([x, x_struct], dim=1)
        x = self.node_fusion(x)

        x, x_e = self.attention_fusion1(x, x_e, edge_index)
        x, x_e = self.attention_fusion2(x, x_e, edge_index)
        x, x_e = self.attention_fusion3(x, x_e, edge_index)

        x_aggr = self.aggr(x[edge_index[0]], edge_index[1])

        x_fused = torch.cat([x_aggr, x_e], dim=1)
        x_fused = self.edge_fusion(x_fused)

        out = self.classifier(x_fused)

        return out
