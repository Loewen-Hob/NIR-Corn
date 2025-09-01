# experiments/cnn_experiment.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

# 定义 1D-CNN 模型
class CNNRegressor(nn.Module):
    def __init__(self, input_length=700, output_dim=4):
        super(CNNRegressor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # 自适应池化到固定长度
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, 700) -> (batch, 1, 700)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

def train_cnn_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """训练 CNN 模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
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
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 早停
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
        elif epoch > 50:
            break
    
    return model

def run_cnn_experiment():
    """运行 CNN 实验"""
    print("🚀 开始 CNN 实验...")
    
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
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = CNNRegressor(input_length=700, output_dim=4)
    
    # 训练模型
    print("🧠 训练 CNN 模型...")
    model = train_cnn_model(model, train_loader, val_loader, epochs=200, lr=0.001, device=device)
    
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
        model_name="CNN",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="CNN",
        save_path="experiments/results/cnn_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "CNN")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "cnn_model.pth"))
    joblib.dump(scaler_X, os.path.join(model_dir, "cnn_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "cnn_scaler_y.pkl"))
    
    print("✅ CNN 实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_cnn_experiment()