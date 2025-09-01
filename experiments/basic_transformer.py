# experiments/basic_transformer.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

class BasicSpectralTransformer(nn.Module):
    """最基本的光谱 Transformer 回归模型"""
    def __init__(self, input_dim=700, output_dim=4, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        # 输入投影层：将 700 维光谱投影到 d_model 维
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码（可学习）
        self.pos_encoding = nn.Parameter(torch.randn(1, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 700)
        batch_size = x.shape[0]
        
        # 1. 输入投影
        x = self.input_projection(x)  # (batch_size, d_model)
        
        # 2. 添加位置编码（扩展为序列形式）
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_encoding.unsqueeze(0)  # 广播加法
        
        # 3. Transformer 编码
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # 4. 全局池化并预测
        x = x.squeeze(1)  # (batch_size, d_model)
        output = self.predictor(x)  # (batch_size, 4)
        
        return output

def train_transformer(model, train_loader, val_loader, epochs=200, lr=0.001, device='cpu'):
    """训练 Transformer 模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    model.to(device)
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
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
        
        # 验证阶段
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
        
        # 打印进度
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "experiments/models/basic_transformer_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model

def run_basic_transformer_experiment():
    """运行基本 Transformer 回归实验"""
    print("🚀 开始基本 Transformer 回归实验...")
    
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
    model = BasicSpectralTransformer(
        input_dim=700, 
        output_dim=4, 
        d_model=128, 
        nhead=8, 
        num_layers=4
    )
    
    # 训练模型
    print("🧠 训练基本 Transformer 模型...")
    model = train_transformer(model, train_loader, val_loader, epochs=200, lr=0.001, device=device)
    
    # 加载最佳模型
    model.load_state_dict(torch.load("experiments/models/basic_transformer_best.pth", map_location=device))
    
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
        model_name="Basic Transformer",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="Basic Transformer",
        save_path="experiments/results/basic_transformer_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "Basic Transformer")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "basic_transformer_model.pth"))
    joblib.dump(scaler_X, os.path.join(model_dir, "basic_transformer_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "basic_transformer_scaler_y.pkl"))
    
    print("✅ 基本 Transformer 回归实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_basic_transformer_experiment()