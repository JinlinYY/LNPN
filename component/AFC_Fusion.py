import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        energy = torch.tanh(self.W(features))
        attention = F.softmax(self.V(energy), dim=1)
        context = attention * features
        return context

def combine_features(image_features, tabular_features):
    attention = Attention(input_dim=image_features.shape[1], output_dim=image_features.shape[1], hidden_dim=64)
    attended_image_features = attention(image_features)
    # Detach tensors before converting to numpy arrays
    attended_image_features_np = attended_image_features.detach().cpu().numpy()
    tabular_features_np = tabular_features.detach().cpu().numpy()
    combined_features = np.concatenate((attended_image_features_np, tabular_features_np), axis=1)
    return combined_features

