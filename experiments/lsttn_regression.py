# experiments/lsttn_regression.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import time

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

class SpectralLSTTN(nn.Module):
    """简化版 LSTTN 用于光谱回归"""
    def __init__(self, input_dim=700, output_dim=4, hidden_dim=96):
        super().__init__()
        
        # 光谱特征提取器（简化版 Transformer）
        self.patch_size = 12  # 将 700 维分成 patches
        self.num_patches = input_dim // self.patch_size
        
        # 输入嵌入
        self.input_embedding = nn.Conv1d(1, hidden_dim, 
                                       kernel_size=self.patch_size, 
                                       stride=self.patch_size)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(self.num_patches, hidden_dim))
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 长期趋势提取器
        self.long_trend_extractor = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 短期趋势提取器（简化版 GCN）
        self.short_trend_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 最终预测头
        self.predictor = nn.Sequential(
            nn.Linear(32 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, 700)
        batch_size = x.shape[0]
        
        # 添加通道维度: (batch, 1, 700)
        x = x.unsqueeze(1)
        
        # 分 patch 并嵌入
        x = self.input_embedding(x)  # (batch, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, hidden_dim)
        
        # 添加位置编码
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # (batch, num_patches, hidden_dim)
        
        # 长期趋势提取
        long_trend = self.long_trend_extractor(
            x.transpose(1, 2)
        ).squeeze(-1)  # (batch, 32)
        
        # 短期趋势提取（使用最后一个 patch）
        short_trend = self.short_trend_extractor(x[:, -1, :])  # (batch, 64)
        
        # 融合特征
        features = torch.cat([long_trend, short_trend], dim=1)
        
        # 预测输出
        output = self.predictor(features)
        
        return output

def train_lsttn_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """训练 LSTTN 模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    model.to(device)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 早停
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # 保存最佳模型
            torch.save(model.state_dict(), "experiments/models/lsttn_best.pth")
        elif epoch > 50:
            break
    
    return model

def run_lsttn_experiment():
    """运行 LSTTN 回归实验"""
    print("🚀 开始 LSTTN 回归实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    # 转换为 PyTorch Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = SpectralLSTTN(input_dim=700, output_dim=4, hidden_dim=96)
    
    # 训练模型
    print("🧠 训练 LSTTN 模型...")
    model = train_lsttn_model(model, train_loader, val_loader, epochs=200, lr=0.001, device=device)
    
    # 加载最佳模型
    model.load_state_dict(torch.load("experiments/models/lsttn_best.pth", map_location=device))
    
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
        model_name="LSTTN",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="LSTTN",
        save_path="experiments/results/lsttn_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "LSTTN")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "lsttn_model.pth"))
    joblib.dump(scaler_X, os.path.join(model_dir, "lsttn_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "lsttn_scaler_y.pkl"))
    
    print("✅ LSTTN 回归实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_lsttn_experiment()