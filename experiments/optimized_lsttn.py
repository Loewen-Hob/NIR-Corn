# experiments/optimized_lsttn_regression.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv
torch.backends.cudnn.enabled = False

class MultiScaleSpectralAttention(nn.Module):
    """多尺度光谱注意力机制"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scales = [3, 7, 15, 31]  # 不同尺度的卷积核
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // len(self.scales), 
                     kernel_size=scale, padding=scale//2)
            for scale in self.scales
        ])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 保存注意力权重用于可视化
        self.attention_weights = None
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x_conv = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # 多尺度特征提取
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x_conv)  # (batch, hidden_dim//4, seq_len)
            scale_features.append(feat)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(scale_features, dim=1)  # (batch, hidden_dim, seq_len)
        multi_scale = multi_scale.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # 自注意力
        attended, attention_weights = self.attention(multi_scale, multi_scale, multi_scale)
        self.attention_weights = attention_weights  # 保存注意力权重
        output = self.norm(attended + multi_scale)
        
        return output

class SpectralPositionalEncoding(nn.Module):
    """光谱专用位置编码"""
    def __init__(self, d_model, max_len=200):  # 改为200
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加光谱频率相关的编码
        spectral_factor = torch.linspace(0, 1, max_len).unsqueeze(1)
        pe = pe * (1 + 0.1 * spectral_factor)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

class AdaptivePatchEmbedding(nn.Module):
    """自适应patch嵌入"""
    def __init__(self, input_dim, embed_dim, patch_sizes=[8, 12, 16]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        
        # 确保能正确分配维度
        if embed_dim % len(patch_sizes) != 0:
            # 如果不能整除，手动分配
            base_dim = embed_dim // len(patch_sizes)
            remainder = embed_dim % len(patch_sizes)
            dims = [base_dim + (1 if i < remainder else 0) for i in range(len(patch_sizes))]
        else:
            dims = [embed_dim // len(patch_sizes)] * len(patch_sizes)
        
        # 多个不同大小的patch嵌入
        self.patch_embeds = nn.ModuleList([
            nn.Conv1d(1, dim, kernel_size=ps, stride=ps)
            for ps, dim in zip(patch_sizes, dims)
        ])
        
        # 自适应权重
        self.patch_weights = nn.Parameter(torch.ones(len(patch_sizes)))
        
    def forward(self, x):
        # x: (batch, 1, 700)
        patch_features = []
        
        for i, (patch_embed, patch_size) in enumerate(zip(self.patch_embeds, self.patch_sizes)):
            # 确保能被patch_size整除
            seq_len = x.size(2)
            trimmed_len = (seq_len // patch_size) * patch_size
            x_trimmed = x[:, :, :trimmed_len]
            
            feat = patch_embed(x_trimmed)  # (batch, dim, num_patches)
            patch_features.append(feat)
        
        # 加权融合不同patch size的特征
        weights = F.softmax(self.patch_weights, dim=0)
        weighted_features = []
        min_patches = min([feat.size(2) for feat in patch_features])
        
        for i, feat in enumerate(patch_features):
            feat_trimmed = feat[:, :, :min_patches]
            weighted_features.append(feat_trimmed * weights[i])
        
        # 拼接特征
        output = torch.cat(weighted_features, dim=1)  # (batch, embed_dim, num_patches)
        return output.transpose(1, 2)  # (batch, num_patches, embed_dim)

class EnhancedTransformerBlock(nn.Module):
    """增强的Transformer块"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # 标准注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 卷积前馈网络（更适合光谱数据）
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_feedforward, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # 保存注意力权重用于可视化
        self.attention_weights = None
        
    def forward(self, x):
        # 自注意力
        attn_out, attention_weights = self.self_attn(x, x, x)
        self.attention_weights = attention_weights  # 保存注意力权重
        x = self.norm1(x + self.dropout(attn_out))
        
        # 卷积前馈
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        ffn_out = self.conv_ffn(x_conv)
        ffn_out = ffn_out.transpose(1, 2)  # (batch, seq_len, d_model)
        
        x = self.norm2(x + ffn_out)
        return x

class OptimizedSpectralLSTTN(nn.Module):
    """优化版 LSTTN 光谱回归模型"""
    def __init__(self, input_dim=700, output_dim=4, hidden_dim=128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. 自适应patch嵌入
        self.patch_embedding = AdaptivePatchEmbedding(
            input_dim, hidden_dim, patch_sizes=[8, 12, 16]
        )
        
        # 2. 光谱专用位置编码
        self.pos_encoding = SpectralPositionalEncoding(hidden_dim, max_len=512) 
        
        # 3. 多尺度光谱注意力
        self.multi_scale_attn = MultiScaleSpectralAttention(hidden_dim, hidden_dim)
        
        # 4. 增强Transformer编码器
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(hidden_dim, nhead=8, dim_feedforward=hidden_dim*4)
            for _ in range(6)  # 增加层数
        ])
        
        # 5. 多路径特征提取器
        # 全局路径：整体趋势
        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 局部路径：关键区域
        self.local_extractor = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2)
        )
        
        # 序列路径：时序信息
        self.sequence_extractor = nn.LSTM(
            hidden_dim, 32, batch_first=True, bidirectional=True
        )
        
        # 6. 自注意力融合
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=64+64+64, num_heads=8, batch_first=True
        )
        
        # 7. 任务特定预测头
        task_dims = {'moisture': 32, 'starch': 32, 'oil': 48, 'protein': 48}  # 根据难度调整
        
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
        
        # 8. 残差连接的最终预测
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
        batch_size = x.shape[0]
        
        # 1. 添加通道维度并进行patch嵌入
        x = x.unsqueeze(1)  # (batch, 1, 700)
        x = self.patch_embedding(x)  # (batch, num_patches, hidden_dim)
        
        # 2. 位置编码
        x = self.pos_encoding(x)
        
        # 3. 多尺度注意力
        x = self.multi_scale_attn(x)
        
        # 4. Transformer编码 (保存每层的注意力权重)
        attention_weights_list = []
        for layer in self.transformer_layers:
            x = layer(x)
            if layer.attention_weights is not None:
                attention_weights_list.append(layer.attention_weights)
        
        # 保存注意力权重用于可视化
        self.attention_weights_list = attention_weights_list
        
        # 5. 多路径特征提取
        x_transpose = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        # 全局特征
        global_feat = self.global_extractor(x_transpose)  # (batch, 64)
        
        # 局部特征
        local_feat = self.local_extractor(x_transpose)  # (batch, 64)
        
        # 序列特征
        seq_feat, _ = self.sequence_extractor(x)  # (batch, seq_len, 64)
        seq_feat = seq_feat.mean(dim=1)  # (batch, 64)
        
        # 6. 特征融合
        combined_feat = torch.cat([global_feat, local_feat, seq_feat], dim=1)  # (batch, 192)
        
        # 自注意力融合
        fused_feat = combined_feat.unsqueeze(1)  # (batch, 1, 192)
        fused_feat, attention_weights_fusion = self.fusion_attention(fused_feat, fused_feat, fused_feat)
        self.fusion_attention_weights = attention_weights_fusion  # 保存融合注意力权重
        fused_feat = fused_feat.squeeze(1)  # (batch, 192)
        
        # 7. 任务特定预测
        task_outputs = []
        task_names = ['moisture', 'starch', 'oil', 'protein']
        
        for task in task_names:
            task_out = self.task_heads[task](fused_feat)
            task_outputs.append(task_out)
        
        task_pred = torch.cat(task_outputs, dim=1)  # (batch, 4)
        
        # 8. 最终预测（残差连接）
        final_pred = self.final_predictor(fused_feat)
        
        # 加权融合任务特定和通用预测
        alpha = 0.7  # 可学习参数
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

def train_optimized_lsttn_model(model, train_loader, val_loader, epochs=150, lr=0.0008, device='cpu'):
    """训练优化版LSTTN模型"""
    
    # 多任务损失函数
    def multi_task_loss(pred, target):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # L1正则化
        l1_loss = F.l1_loss(pred, target)
        
        # 任务平衡损失
        task_losses = []
        for i in range(target.size(1)):
            task_loss = F.mse_loss(pred[:, i], target[:, i])
            task_losses.append(task_loss)
        
        # 动态权重调整
        task_weights = torch.tensor([1.2, 1.0, 1.5, 2.0]).to(device)  # 根据任务难度调整
        weighted_loss = sum(w * l for w, l in zip(task_weights, task_losses))
        
        return 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * weighted_loss
    
    # 优化器设置 - 增加学习率和权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # 学习率调度器 - 调整参数
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy='cos'
    )
    
    model.to(device)
    best_loss = float('inf')
    patience = 30  # 增加耐心
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = multi_task_loss(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = multi_task_loss(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "experiments/models/optimized_lsttn_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 80:  # 增加最小训练轮数
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model

def run_optimized_lsttn_experiment():
    """运行优化版LSTTN回归实验"""
    print("🚀 开始优化版 LSTTN 回归实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # def augment_spectral_data(X, y, noise_level=0.005, shift_range=2):  # 降低增强强度
    #     """轻量级光谱数据增强"""
    #     augmented_X, augmented_y = [], []
        
    #     for i in range(len(X)):
    #         # 原始数据
    #         augmented_X.append(X[i])
    #         augmented_y.append(y[i])
            
    #         # 添加轻微噪声
    #         noise = np.random.normal(0, noise_level, X[i].shape)
    #         augmented_X.append(X[i] + noise)
    #         augmented_y.append(y[i])
            
    #         # 轻微光谱偏移
    #         shift = np.random.randint(-shift_range, shift_range+1)
    #         if shift != 0:
    #             shifted_x = np.roll(X[i], shift)
    #             augmented_X.append(shifted_x)
    #             augmented_y.append(y[i])
        
    #     return np.array(augmented_X), np.array(augmented_y)
    
    # 应用轻量级数据增强
    # X_train_aug, y_train_scaled_aug = augment_spectral_data(X_train, y_train_scaled)

    X_train_aug, y_train_scaled_aug = X_train, y_train_scaled
    print(f"📈 轻量级数据增强后: {X_train_aug.shape[0]} 样本")
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_aug)
    y_train_tensor = torch.FloatTensor(y_train_scaled_aug)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)  # 减小batch size
    
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建优化模型 - 调整超参数
    model = OptimizedSpectralLSTTN(input_dim=700, output_dim=4, hidden_dim=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params:,}")
    
    # 训练模型 - 调整训练参数
    print("🧠 训练优化版 LSTTN 模型...")
    model = train_optimized_lsttn_model(
        model, train_loader, val_loader, 
        epochs=250, lr=0.001, device=device  # 增加训练轮数，提高学习率
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load("experiments/models/optimized_lsttn_best.pth", map_location=device))
    
    # 预测
    print("🔮 进行预测...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    
    # 评估
    metrics = evaluate_model(
        y_test_orig, y_pred_orig, 
        model_name="Optimized-LSTTN",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="Optimized LSTTN",
        save_path="experiments/results/optimized_lsttn_predictions.png"
    )
    
    # 注意力可视化
    print("🎨 生成注意力热力图...")
    # 选择几个测试样本进行可视化
    sample_indices = [0, 1, 2, 5, 10]  # 选择前几个样本
    for idx in sample_indices:
        if idx < len(X_test):
            visualize_attention_weights(model, X_test[idx], idx, device=device)  # 传递设备参数
    
    print("📊 注意力热力图已保存到: experiments/results/attention_maps/")
    
    # 保存结果
    save_results_to_csv(metrics, "Optimized-LSTTN")
    
    # 保存模型
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "optimized_lsttn_model.pth"))
    joblib.dump(scaler_X, os.path.join(model_dir, "optimized_lsttn_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "optimized_lsttn_scaler_y.pkl"))
    
    print("✅ 优化版 LSTTN 回归实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_optimized_lsttn_experiment()