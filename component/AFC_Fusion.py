import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# def combine_features(image_features, tabular_features):
#     return np.concatenate((image_features, tabular_features), axis=1)

# 实验用的这个
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

# class Attention(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, num_heads=1):
#         super(Attention, self).__init__()
#         self.W = nn.Linear(input_dim, hidden_dim * num_heads)
#         self.V = nn.Linear(hidden_dim * num_heads, output_dim)
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#
#     def forward(self, features):
#         energy = torch.tanh(self.W(features))
#         attention = F.softmax(self.V(energy), dim=1)
#         context = attention * features
#         return context
#
# def combine_features(image_features, tabular_features):
#     attention = Attention(input_dim=image_features.shape[1], output_dim=image_features.shape[1], hidden_dim=64, num_heads=4)
#     attended_image_features = attention(image_features)
#     # Normalize and combine
#     combined_features = torch.cat((attended_image_features, tabular_features), dim=1)
#     return combined_features.detach().cpu().numpy()



###########################多头注意力机制####################################################################################
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, query, key, value):
#         attn_output, _ = self.multihead_attn(query, key, value)
#         return self.output_proj(attn_output)
#
# class FeatureFusion(nn.Module):
#     def __init__(self, image_dim, tabular_dim, embed_dim, num_heads):
#         super(FeatureFusion, self).__init__()
#         self.image_proj = nn.Linear(image_dim, embed_dim)
#         self.tabular_proj = nn.Linear(tabular_dim, embed_dim)
#         self.multihead_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, image_features, tabular_features):
#         image_proj = self.image_proj(image_features)
#         tabular_proj = self.tabular_proj(tabular_features)
#         combined = torch.cat((image_proj.unsqueeze(1), tabular_proj.unsqueeze(1)), dim=1)
#         attended_features = self.multihead_attention(combined, combined, combined).mean(dim=1)
#         return self.output_proj(attended_features)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = FeatureFusion(image_dim=image_features.shape[1], tabular_dim=tabular_features.shape[1], embed_dim=64, num_heads=8)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()


# ############################外积矩阵####################################################################################
# import torch
# import torch.nn as nn
#
# class OuterProductFusion(nn.Module):
#     def __init__(self):
#         super(OuterProductFusion, self).__init__()
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product_matrix = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1))
#         return outer_product_matrix.view(batch_size, -1)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = OuterProductFusion()
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()


# ############################双线性池化####################################################################################
# import torch
# import torch.nn as nn
#
# class BilinearPooling(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(BilinearPooling, self).__init__()
#         self.fc = nn.Linear(input_dim1 * input_dim2, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1)).view(batch_size, -1)
#         return self.fc(outer_product)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = BilinearPooling(input_dim1=image_features.shape[1], input_dim2=tabular_features.shape[1], output_dim=128)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()

# ############################共注意力机制##################################################################################
# import torch
# import torch.nn as nn
#
# class CoAttention(nn.Module):
#     def __init__(self, image_dim, tabular_dim, output_dim):
#         super(CoAttention, self).__init__()
#         # Calculate projection dimensions
#         self.image_fc = nn.Linear(image_dim, output_dim)
#         self.tabular_fc = nn.Linear(tabular_dim, output_dim)
#         self.output_proj = nn.Linear(output_dim * output_dim, output_dim)  # Update this line
#
#     def forward(self, image_features, tabular_features):
#         # Project image and tabular features
#         image_proj = self.image_fc(image_features)
#         tabular_proj = self.tabular_fc(tabular_features)
#
#         # Compute attention matrix and flatten
#         attention_matrix = torch.bmm(image_proj.unsqueeze(2), tabular_proj.unsqueeze(1))
#         co_attention = attention_matrix.view(image_features.size(0), -1)
#
#         # Apply linear projection
#         return self.output_proj(co_attention)

# def combine_features(image_features, tabular_features):
#     # Ensure dimensions match with the CoAttention implementation
#     fusion_model = CoAttention(image_dim=image_features.shape[1], tabular_dim=tabular_features.shape[1], output_dim=128)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()
