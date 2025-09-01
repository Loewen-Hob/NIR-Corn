# experiments/visualize_attention.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import load_data
torch.backends.cudnn.enabled = False

class MultiScaleSpectralAttention(nn.Module):
    """多尺度光谱注意力机制"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scales = [3, 7, 15, 31]
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // len(self.scales), 
                     kernel_size=scale, padding=scale//2)
            for scale in self.scales
        ])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_weights = None
        
    def forward(self, x):
        x_conv = x.transpose(1, 2)
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x_conv)
            scale_features.append(feat)
        multi_scale = torch.cat(scale_features, dim=1)
        multi_scale = multi_scale.transpose(1, 2)
        attended, attention_weights = self.attention(multi_scale, multi_scale, multi_scale)
        self.attention_weights = attention_weights
        output = self.norm(attended + multi_scale)
        return output

class SpectralPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        from math import exp, log
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        spectral_factor = torch.linspace(0, 1, max_len).unsqueeze(1)
        pe = pe * (1 + 0.1 * spectral_factor)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_sizes=[8, 12, 16]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        if embed_dim % len(patch_sizes) != 0:
            base_dim = embed_dim // len(patch_sizes)
            remainder = embed_dim % len(patch_sizes)
            dims = [base_dim + (1 if i < remainder else 0) for i in range(len(patch_sizes))]
        else:
            dims = [embed_dim // len(patch_sizes)] * len(patch_sizes)
        self.patch_embeds = nn.ModuleList([
            nn.Conv1d(1, dim, kernel_size=ps, stride=ps)
            for ps, dim in zip(patch_sizes, dims)
        ])
        self.patch_weights = nn.Parameter(torch.ones(len(patch_sizes)))
        
    def forward(self, x):
        patch_features = []
        for i, (patch_embed, patch_size) in enumerate(zip(self.patch_embeds, self.patch_sizes)):
            seq_len = x.size(2)
            trimmed_len = (seq_len // patch_size) * patch_size
            x_trimmed = x[:, :, :trimmed_len]
            feat = patch_embed(x_trimmed)
            patch_features.append(feat)
        weights = torch.softmax(self.patch_weights, dim=0)
        weighted_features = []
        min_patches = min([feat.size(2) for feat in patch_features])
        for i, feat in enumerate(patch_features):
            feat_trimmed = feat[:, :, :min_patches]
            weighted_features.append(feat_trimmed * weights[i])
        output = torch.cat(weighted_features, dim=1)
        return output.transpose(1, 2)

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_feedforward, d_model, 1),
            nn.Dropout(dropout)
        )
        self.attention_weights = None
        
    def forward(self, x):
        attn_out, attention_weights = self.self_attn(x, x, x)
        self.attention_weights = attention_weights
        x = self.norm1(x + self.dropout(attn_out))
        x_conv = x.transpose(1, 2)
        ffn_out = self.conv_ffn(x_conv)
        ffn_out = ffn_out.transpose(1, 2)
        x = self.norm2(x + ffn_out)
        return x

class OptimizedSpectralLSTTN(nn.Module):
    def __init__(self, input_dim=700, output_dim=4, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embedding = AdaptivePatchEmbedding(input_dim, hidden_dim, patch_sizes=[8, 12, 16])
        self.pos_encoding = SpectralPositionalEncoding(hidden_dim, max_len=512)
        self.multi_scale_attn = MultiScaleSpectralAttention(hidden_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(hidden_dim, nhead=8, dim_feedforward=hidden_dim*4)
            for _ in range(6)
        ])
        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.local_extractor = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2)
        )
        self.sequence_extractor = nn.LSTM(hidden_dim, 32, batch_first=True, bidirectional=True)
        self.fusion_attention = nn.MultiheadAttention(embed_dim=64+64+64, num_heads=8, batch_first=True)
        task_dims = {'moisture': 32, 'starch': 32, 'oil': 48, 'protein': 48}
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(64+64+64, dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(dim, dim//2),
                nn.ReLU(),
                nn.Linear(dim//2, 1)
            ) for task, dim in task_dims.items()
        })
        self.final_predictor = nn.Sequential(
            nn.Linear(64+64+64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.pos_encoding(x)
        x = self.multi_scale_attn(x)
        attention_weights_list = []
        for layer in self.transformer_layers:
            x = layer(x)
            if layer.attention_weights is not None:
                attention_weights_list.append(layer.attention_weights)
        self.attention_weights_list = attention_weights_list
        x_transpose = x.transpose(1, 2)
        global_feat = self.global_extractor(x_transpose)
        local_feat = self.local_extractor(x_transpose)
        seq_feat, _ = self.sequence_extractor(x)
        seq_feat = seq_feat.mean(dim=1)
        combined_feat = torch.cat([global_feat, local_feat, seq_feat], dim=1)
        fused_feat = combined_feat.unsqueeze(1)
        fused_feat, attention_weights_fusion = self.fusion_attention(fused_feat, fused_feat, fused_feat)
        self.fusion_attention_weights = attention_weights_fusion
        fused_feat = fused_feat.squeeze(1)
        task_outputs = []
        task_names = ['moisture', 'starch', 'oil', 'protein']
        for task in task_names:
            task_out = self.task_heads[task](fused_feat)
            task_outputs.append(task_out)
        task_pred = torch.cat(task_outputs, dim=1)
        final_pred = self.final_predictor(fused_feat)
        alpha = 0.7
        output = alpha * task_pred + (1 - alpha) * final_pred
        return output

def visualize_attention_weights(model, X_sample, sample_idx=0, save_dir="/ssd1/zhanghongbo04/002/project/NIR-Corn/experiments/experiments/results/attention_maps", device='cpu'):
    """可视化注意力权重"""
    os.makedirs(save_dir, exist_ok=True)
    torch.backends.cudnn.enabled = False

    model.eval()
    model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).to(device)
        _ = model(X_tensor)
        
        # 可视化多尺度注意力
        if hasattr(model.multi_scale_attn, 'attention_weights') and model.multi_scale_attn.attention_weights is not None:
            attn_weights = model.multi_scale_attn.attention_weights
            print(f"✅ 多尺度注意力权重形状: {attn_weights.shape}")
            print(f"注意力权重类型: {type(attn_weights)}")
            
            # 检查不同的维度情况
            if isinstance(attn_weights, torch.Tensor):
                if len(attn_weights.shape) == 4:  # (batch, heads, seq_len, seq_len)
                    attn_map = attn_weights[0, 0, :, :].cpu().numpy()
                elif len(attn_weights.shape) == 3:  # (heads, seq_len, seq_len)
                    attn_map = attn_weights[0, :, :].cpu().numpy()
                elif len(attn_weights.shape) == 2:  # (seq_len, seq_len)
                    attn_map = attn_weights.cpu().numpy()
                else:
                    print(f"未知的注意力权重形状: {attn_weights.shape}")
                    return
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn_map, cmap='viridis', cbar=True)
                plt.title(f'Multi-Scale Attention Weights - Sample {sample_idx}')
                plt.xlabel('Key Positions')
                plt.ylabel('Query Positions')
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'multi_scale_attention_sample_{sample_idx}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ 保存多尺度注意力图: {save_path}")
            else:
                print("注意力权重不是Tensor类型")
        
        # 可视化Transformer各层注意力
        if hasattr(model, 'attention_weights_list') and model.attention_weights_list:
            for layer_idx, attn_weights in enumerate(model.attention_weights_list):
                print(f"Transformer层 {layer_idx+1} 注意力权重形状: {attn_weights.shape}")
                if isinstance(attn_weights, torch.Tensor):
                    if len(attn_weights.shape) == 4:  # (batch, heads, seq_len, seq_len)
                        attn_map = attn_weights[0, 0, :, :].cpu().numpy()
                    elif len(attn_weights.shape) == 3:  # (heads, seq_len, seq_len)
                        attn_map = attn_weights[0, :, :].cpu().numpy()
                    elif len(attn_weights.shape) == 2:  # (seq_len, seq_len)
                        attn_map = attn_weights.cpu().numpy()
                    else:
                        print(f"未知的注意力权重形状: {attn_weights.shape}")
                        continue
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(attn_map, cmap='viridis', cbar=True)
                    plt.title(f'Transformer Layer {layer_idx+1} Attention Weights - Sample {sample_idx}')
                    plt.xlabel('Key Positions')
                    plt.ylabel('Query Positions')
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'transformer_layer_{layer_idx+1}_attention_sample_{sample_idx}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ 保存Transformer层{layer_idx+1}注意力图: {save_path}")
                else:
                    print("注意力权重不是Tensor类型")

def load_model_and_visualize():
    """加载模型并生成注意力可视化"""
    print("🎨 开始注意力可视化...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 加载数据
    print("📊 加载测试数据...")
    (_, X_test, _, _, _, _, _, y_test_orig) = load_data()
    print(f"✅ 测试集大小: {X_test.shape}")
    
    # 创建模型
    print("🧠 创建模型...")
    model = OptimizedSpectralLSTTN(input_dim=700, output_dim=4, hidden_dim=128)
    
    # 加载训练好的模型权重
    model_path = "/ssd1/zhanghongbo04/002/project/NIR-Corn/experiments/experiments/models/optimized_lsttn_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型: {model_path}")
    else:
        print(f"❌ 未找到模型文件: {model_path}")
        return
    
    # 选择样本进行可视化
    print("🎨 生成注意力热力图...")
    sample_indices = [0, 1, 2, 5, 10, 15, 20]  # 选择多个样本
    
    for idx in sample_indices:
        if idx < len(X_test):
            print(f"  处理样本 {idx}...")
            visualize_attention_weights(model, X_test[idx], idx, device=device)
    
    print("✅ 注意力可视化完成!")
    print("📁 结果保存在: experiments/results/attention_maps/")

if __name__ == "__main__":
    load_model_and_visualize()