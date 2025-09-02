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
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv
torch.backends.cudnn.enabled = False
print("cuDNN 已禁用，使用替代实现")

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
        # self.attention_weights = None
        
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
        # self.attention_weights = attention_weights  # 保存注意力权重
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
        # self.attention_weights = None
        
    def forward(self, x):
        # 自注意力
        attn_out, attention_weights = self.self_attn(x, x, x)
        # self.attention_weights = attention_weights  # 保存注意力权重
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
            # if layer.attention_weights is not None:
            #     attention_weights_list.append(layer.attention_weights)
        
        # 保存注意力权重用于可视化
        # self.attention_weights_list = attention_weights_list
        
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
        # self.fusion_attention_weights = attention_weights_fusion  # 保存融合注意力权重
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

class GradientInterpreter:
    """
    基于梯度的模型可解释性分析工具
    提供输入特征重要性分析和中间层重要性分析
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()  # 设置为评估模式
    
    def compute_feature_importance(self, X, target_component=0):
        """
        计算输入特征（波长）对特定输出成分的重要性
        target_component: 0=水分, 1=淀粉, 2=油脂, 3=蛋白质
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        else:
            X = X.to(self.device)
        
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        # 临时切换到训练模式以确保梯度计算
        original_mode = self.model.training
        self.model.train()
        
        try:
            # 清除之前的梯度
            self.model.zero_grad()
            
            # 重新创建需要梯度的输入
            X_with_grad = X.clone().detach().requires_grad_(True)
            
            print("=== 调试信息 ===")
            print("Input requires_grad:", X_with_grad.requires_grad)
            
            # 逐层检查梯度传播
            with torch.set_grad_enabled(True):
                # 1. 检查patch embedding
                x1 = X_with_grad.unsqueeze(1)
                print("After unsqueeze requires_grad:", x1.requires_grad)
                
                x2 = self.model.patch_embedding(x1)
                print("After patch_embedding requires_grad:", x2.requires_grad)
                
                # 2. 检查位置编码
                x3 = self.model.pos_encoding(x2)
                print("After pos_encoding requires_grad:", x3.requires_grad)
                
                # 3. 检查多尺度注意力
                x4 = self.model.multi_scale_attn(x3)
                print("After multi_scale_attn requires_grad:", x4.requires_grad)
                
                # 4. 检查transformer层
                x5 = x4
                for i, layer in enumerate(self.model.transformer_layers):
                    x5 = layer(x5)
                    print(f"After transformer layer {i} requires_grad:", x5.requires_grad)
                    if not x5.requires_grad:
                        print(f"❌ 问题出现在第 {i} 层 Transformer")
                        break
                
                # 5. 继续检查后续层...
                output = self.model(X_with_grad)
                print("Final output requires_grad:", output.requires_grad)
                
                if not output.requires_grad:
                    print("❌ 梯度在模型中某处中断了")
                    return None, None
                
                # 选择特定目标成分
                target_output = output[:, target_component]
                
                # 计算梯度
                gradients = torch.autograd.grad(
                    outputs=target_output,
                    inputs=X_with_grad,
                    grad_outputs=torch.ones_like(target_output),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]
                
                importance = torch.abs(gradients)
                
                return importance.detach().cpu().numpy(), output.detach().cpu().numpy()
        
        except Exception as e:
            print(f"❌ 计算梯度时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
        finally:
            # 恢复原来的模式
            self.model.train(original_mode)
    
    def compute_integrated_gradients(self, X, target_component=0, steps=50):
        """
        计算积分梯度（Integrated Gradients），更稳定的特征重要性评估
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        # 基线输入（可以是零向量或随机噪声）
        baseline = torch.zeros_like(X)
        
        # 生成从基线到输入之间的路径
        scaled_inputs = [baseline + (float(i) / steps) * (X - baseline) for i in range(0, steps + 1)]
        scaled_inputs = torch.stack(scaled_inputs)
        
        # 计算每个路径点的梯度
        gradients = []
        for input_step in scaled_inputs:
            input_step.requires_grad_(True)
            output = self.model(input_step)
            target_output = output[:, target_component]
            
            self.model.zero_grad()
            target_output.backward(torch.ones_like(target_output))
            
            grad = input_step.grad.detach().cpu().numpy()
            gradients.append(grad)
            
        # 计算积分梯度近似（黎曼和）
        gradients = np.array(gradients)
        avg_gradients = np.mean(gradients[:-1], axis=0)
        integrated_gradients = (X.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_gradients
        
        return integrated_gradients, output.detach().cpu().numpy()

def visualize_feature_importance(importance_scores, 
                                original_spectrum, 
                                sample_idx, 
                                epoch, 
                                target_name,
                                save_dir="experiments/results/feature_importance"):
    """
    可视化特征重要性并将结果保存为图片
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保重要性分数和原始光谱长度一致
    assert len(importance_scores) == len(original_spectrum), \
        f"重要性分数长度({len(importance_scores)})与光谱长度({len(original_spectrum)})不匹配"
    
    wavelengths = np.arange(len(original_spectrum))
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 绘制原始光谱
    ax1.plot(wavelengths, original_spectrum, 'b-', alpha=0.7, linewidth=1)
    ax1.set_title(f'Original NIR Spectrum - Sample {sample_idx}')
    ax1.set_xlabel('Wavelength Index')
    ax1.set_ylabel('Absorbance')
    ax1.grid(True, alpha=0.3)
    
    # 绘制特征重要性
    ax2.bar(wavelengths, importance_scores, alpha=0.6, color='red', width=1.0)
    ax2.set_title(f'Feature Importance for {target_name} Prediction - Epoch {epoch}')
    ax2.set_xlabel('Wavelength Index')
    ax2.set_ylabel('Importance Score (|dOutput / dInput|)')
    ax2.grid(True, alpha=0.3)
    
    # 标识最重要的5个波长点
    top5_indices = np.argsort(importance_scores)[-5:][::-1]
    for i, idx in enumerate(top5_indices):
        ax2.annotate(f'Top {i+1}', 
                    xy=(idx, importance_scores[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                    fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{target_name}_importance_sample_{sample_idx}_epoch_{epoch}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_component_comparison(importance_scores_list, component_names, 
                                  sample_idx, epoch, 
                                  save_dir="experiments/results/component_comparison"):
    """
    可视化不同成分的特征重要性对比
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    # 绘制每个成分的重要性曲线
    for i, (scores, name) in enumerate(zip(importance_scores_list, component_names)):
        plt.plot(scores, alpha=0.7, label=name, linewidth=1.5)
    
    plt.title(f'Feature Importance Comparison Across Components - Sample {sample_idx} - Epoch {epoch}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    filename = f"component_comparison_sample_{sample_idx}_epoch_{epoch}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_attention_weights(model, X_sample, sample_idx=0, save_dir="experiments/results/attention_maps", device='cpu'):
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

def generate_interpretability_report(model, X_test, y_test, scaler_y, device='cpu'):
    """
    生成最终的可解释性分析报告
    """
    print("📊 生成最终可解释性分析报告...")
    
    interpreter = GradientInterpreter(model, device)
    component_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    
    # 创建报告目录
    report_dir = "experiments/results/interpretability_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 选择几个测试样本进行详细分析
    sample_indices = [0, 5, 10, 15, 20]
    
    # 分析每个样本
    for i, idx in enumerate(sample_indices):
        if idx < len(X_test):
            sample = X_test[idx]
            true_values = y_test[idx]
            
            plt.figure(figsize=(16, 12))
            
            # 获取预测值
            with torch.no_grad():
                sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                prediction = model(sample_tensor).cpu().numpy()
                prediction = scaler_y.inverse_transform(prediction.reshape(1, -1)).squeeze()
            
            # 计算每个成分的重要性
            importance_maps = []
            for comp_idx, comp_name in enumerate(component_names):
                importance, _ = interpreter.compute_feature_importance(sample, comp_idx)
                importance_maps.append(importance.squeeze())
            
            # 创建综合可视化
            for comp_idx, comp_name in enumerate(component_names):
                plt.subplot(2, 2, comp_idx + 1)
                
                # 绘制原始光谱和重要性叠加
                wavelengths = np.arange(len(sample))
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # 原始光谱
                ax1.plot(wavelengths, sample, 'b-', alpha=0.7, label='Spectrum')
                ax1.set_xlabel('Wavelength Index')
                ax1.set_ylabel('Absorbance', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                
                # 重要性分数
                ax2.fill_between(wavelengths, 0, importance_maps[comp_idx], 
                               alpha=0.3, color='r', label='Importance')
                ax2.set_ylabel('Importance Score', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                plt.title(f'{comp_name}\nTrue: {true_values[comp_idx]:.3f}, Pred: {prediction[comp_idx]:.3f}')
                
                # 添加图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            plt.suptitle(f'Feature Importance Analysis - Sample {idx}\n', fontsize=16)
            plt.subplots_adjust(top=0.93)
            
            # 保存综合报告
            report_path = os.path.join(report_dir, f'interpretability_report_sample_{idx}.png')
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 样本 {idx} 的可解释性报告已保存: {report_path}")
    
    # 生成全局特征重要性总结
    plt.figure(figsize=(14, 10))
    
    # 计算所有测试样本的平均重要性
    avg_importances = np.zeros((len(component_names), X_test.shape[1]))
    
    for comp_idx in range(len(component_names)):
        sample_importances = []
        for i in range(min(50, len(X_test))):  # 使用前50个样本计算平均
            importance, _ = interpreter.compute_feature_importance(X_test[i], comp_idx)
            sample_importances.append(importance.squeeze())
        
        avg_importances[comp_idx] = np.mean(sample_importances, axis=0)
    
    # 绘制全局重要性
    for comp_idx, comp_name in enumerate(component_names):
        plt.subplot(2, 2, comp_idx + 1)
        plt.plot(avg_importances[comp_idx], 'r-', alpha=0.8)
        plt.title(f'Global Feature Importance - {comp_name}')
        plt.xlabel('Wavelength Index')
        plt.ylabel('Average Importance Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    global_report_path = os.path.join(report_dir, 'global_feature_importance.png')
    plt.savefig(global_report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 全局特征重要性报告已保存: {global_report_path}")
    print("📋 可解释性分析报告生成完成!")

def train_optimized_lsttn_model(model, train_loader, val_loader, X_train_tensor, epochs=150, lr=0.0008, device='cpu'):
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
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    
    model.to(device)
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # 创建可解释性分析器
    interpreter = GradientInterpreter(model, device)
    
    # 选择几个代表性样本进行分析
    analysis_samples = []
    sample_indices = [0, 1, 2, 5, 10]  # 选择不同的样本进行分析
    for idx in sample_indices:
        if idx < len(X_train_tensor):
            analysis_samples.append(X_train_tensor[idx])
    
    component_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    
    for epoch in range(epochs):
        # 训练循环
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
        
        # 验证循环
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = multi_task_loss(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 每20个epoch进行一次可解释性分析
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # 进行可解释性分析
            model.eval()
            with torch.no_grad():
                # 对每个分析样本进行分析
                for sample_idx, sample in enumerate(analysis_samples):
                    if sample_idx < 3:  # 只分析前3个样本以减少计算量
                        # 分析每个成分的重要性
                        sample_on_device = sample.to(device)

                        all_importances = []
                        
                        for comp_idx, comp_name in enumerate(component_names):
                            # 计算特征重要性
                            importance, prediction = interpreter.compute_feature_importance(
                                sample_on_device, target_component=comp_idx
                            )
                            
                            # 可视化并保存
                            save_path = visualize_feature_importance(
                                importance.squeeze(),
                                sample.cpu().numpy().squeeze(),
                                sample_idx,
                                epoch,
                                comp_name,
                                save_dir="experiments/results/feature_importance"
                            )
                            
                            all_importances.append(importance.squeeze())
                        
                        # 可视化不同成分的重要性对比
                        comp_save_path = visualize_component_comparison(
                            all_importances,
                            component_names,
                            sample_idx,
                            epoch,
                            save_dir="experiments/results/component_comparison"
                        )
                        
                        print(f"✅ 可解释性分析结果已保存: {save_path}, {comp_save_path}")
        
        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "experiments/models/optimized_lsttn_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 50:
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
    
    # 数据增强函数 (可根据需要启用)
    def augment_spectral_data(X, y, noise_level=0.005, shift_range=2):
        """轻量级光谱数据增强"""
        augmented_X, augmented_y = [], []
        
        for i in range(len(X)):
            # 原始数据
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # 添加轻微噪声
            noise = np.random.normal(0, noise_level, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])
            
            # 轻微光谱偏移
            shift = np.random.randint(-shift_range, shift_range+1)
            if shift != 0:
                shifted_x = np.roll(X[i], shift)
                augmented_X.append(shifted_x)
                augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    # 应用轻量级数据增强 (可选)
    # X_train_aug, y_train_scaled_aug = augment_spectral_data(X_train, y_train_scaled)
    # print(f"📈 轻量级数据增强后: {X_train_aug.shape[0]} 样本")
    
    # 不使用数据增强
    X_train_aug, y_train_scaled_aug = X_train, y_train_scaled
    print(f"📈 使用原始数据: {X_train_aug.shape[0]} 样本")
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_aug)
    y_train_tensor = torch.FloatTensor(y_train_scaled_aug)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建优化模型
    model = OptimizedSpectralLSTTN(input_dim=700, output_dim=4, hidden_dim=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params:,}")
    
    # 训练模型
    print("🧠 训练优化版 LSTTN 模型...")
    model = train_optimized_lsttn_model(
        model, train_loader, val_loader, X_train_tensor,
        epochs=250, lr=0.001, device=device
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
            visualize_attention_weights(model, X_test[idx], idx, device=device)
    
    print("📊 注意力热力图已保存到: experiments/results/attention_maps/")
    
    # 生成最终的可解释性分析报告
    generate_interpretability_report(
        model, 
        X_test, 
        y_test_orig,  # 使用逆变换后的原始值
        scaler_y, 
        device=device
    )
    
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