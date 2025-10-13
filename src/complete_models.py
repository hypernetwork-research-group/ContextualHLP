import torch.nn as nn
import torch
from torch_geometric.nn import SetTransformerAggregation, MinAggregation
from torch_geometric.nn.norm import GraphNorm
from .layers import *

class StructureModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.LeakyReLU()):
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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.LeakyReLU()):
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
        self.node_fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.norm_fusion = GraphNorm(hidden_channels)
        
        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_struct = self.structural_refinement(x_struct, edge_index)
        x = self.semantic_refinement(x, edge_index)
        
        x = torch.cat([x, x_struct], dim=1)
        x = self.node_fusion(x)
        x = self.activation(x)
        x = self.norm_fusion(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])

        x = self.classifier(x)
        return x

class NodeAndHyperedges(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.LeakyReLU()):
        super(NodeAndHyperedges, self).__init__()
        
        self.activation = activation
        self.semantic_hyperedge_refinement = SemanticHyperedgeRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)

        self.edge_fusion = nn.Sequential(
            GraphNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            GraphNorm(hidden_channels),
        )
        
        self.attention_fusion = NodeHyperedgeAttention(hidden_channels, hidden_channels)

        self.aggr = MinAggregation()
        
        self.classifier = Classifier(hidden_channels, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_e = self.semantic_hyperedge_refinement(x_e)
        
        x = self.semantic_refinement(x, edge_index)
        x = self.attention_fusion(x, x_e, edge_index)


        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = torch.cat([x, x_e], dim=1)
        x = self.edge_fusion(x)
        x = self.activation(x)

        x = self.classifier(x)
        return x
    
class FullModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3, activation: nn.Module = nn.Tanh()):
        super(FullModel, self).__init__()
        
        self.activation = activation
        self.structural_refinement = StructuralFeatureRefiner(768, hidden_channels, out_channels, dropout, activation)
        self.semantic_refinement = SemanticFeatureRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        
        self.node_fusion = nn.Sequential(
            GraphNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            GraphNorm(hidden_channels),
        )

        self.semantic_hyperedge_refinement = SemanticHyperedgeRefiner(in_channels, hidden_channels, out_channels, dropout, activation)
        
        self.attention_fusion = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.aggr = SetTransformerAggregation(hidden_channels, heads=4)
        
        self.classifier = Classifier(hidden_channels * 2, hidden_channels, out_channels, dropout)
    
    def forward(self, x, x_struct, x_e, edge_index):
        x_e = self.semantic_hyperedge_refinement(x_e)
        x = self.semantic_refinement(x, edge_index)
        
        x = self.attention_fusion(x, x_e, edge_index)
        
        x_struct = self.structural_refinement(x_struct, edge_index)
        x = self.node_fusion(torch.cat([x, x_struct], dim=1))

        x = self.aggr(x[edge_index[0]], edge_index[1])

        x = self.classifier(torch.cat([x, x_e], dim=1))
        return x