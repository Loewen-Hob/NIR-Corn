# experiments/utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, target_names=None, scaler_y=None, model_name="Model"):
    """
    评估回归模型性能
    
    Parameters:
    - y_true: 真实值 (n_samples, n_targets)
    - y_pred: 预测值 (n_samples, n_targets)
    - target_names: 目标变量名称列表
    - scaler_y: y的归一化器（用于还原单位）
    - model_name: 模型名称
    
    Returns:
    - metrics: 包含各项指标的字典
    """
    if target_names is None:
        target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    
    # 如果提供了归一化器，还原到原始单位
    if scaler_y is not None:
        y_true_orig = scaler_y.inverse_transform(y_true)
        y_pred_orig = scaler_y.inverse_transform(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    metrics = {}
    print(f"🎯 {model_name} 评估结果:")
    print("-" * 50)
    
    total_r2 = 0
    total_rmse = 0
    
    for i, name in enumerate(target_names):
        # 计算指标
        r2 = r2_score(y_true_orig[:, i], y_pred_orig[:, i])
        rmse = np.sqrt(mean_squared_error(y_true_orig[:, i], y_pred_orig[:, i]))
        mae = mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i])
        
        metrics[name] = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        total_r2 += r2
        total_rmse += rmse
        
        print(f"  {name:8s}: R²={r2:6.4f}, RMSE={rmse:6.4f}, MAE={mae:6.4f}")
    
    avg_r2 = total_r2 / len(target_names)
    avg_rmse = total_rmse / len(target_names)
    metrics['Average'] = {
        'R2': avg_r2,
        'RMSE': avg_rmse
    }
    
    print("-" * 50)
    print(f"  {'Average':8s}: R²={avg_r2:6.4f}, RMSE={avg_rmse:6.4f}")
    print()
    
    return metrics

def plot_predictions(y_true, y_pred, target_names=None, model_name="Model", save_path=None):
    """
    画出预测值 vs 真实值的散点图
    
    Parameters:
    - y_true: 真实值
    - y_pred: 预测值
    - target_names: 目标变量名称
    - model_name: 模型名称
    - save_path: 保存路径
    """
    if target_names is None:
        target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    
    plt.figure(figsize=(15, 12))
    
    for i, name in enumerate(target_names):
        plt.subplot(2, 2, i+1)
        
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # 画散点图
        plt.scatter(y_t, y_p, alpha=0.6, s=20)
        
        # 画理想线
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        
        # 计算 R²
        r2 = r2_score(y_t, y_p)
        
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{name} (R²={r2:.4f})')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Predictions vs True Values', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 预测图已保存: {save_path}")
    
    plt.show()

def save_results_to_csv(metrics, model_name, output_path="experiments/results/comparison.csv"):
    """
    将结果保存到 CSV 文件中
    
    Parameters:
    - metrics: 评估指标字典
    - model_name: 模型名称
    - output_path: 输出路径
    """
    # 准备数据
    row_data = {'Model': model_name}
    
    for target, scores in metrics.items():
        if target != 'Average':
            row_data[f'{target}_R2'] = scores['R2']
            row_data[f'{target}_RMSE'] = scores['RMSE']
            row_data[f'{target}_MAE'] = scores['MAE']
    
    # 添加平均值
    row_data['Avg_R2'] = metrics['Average']['R2']
    row_data['Avg_RMSE'] = metrics['Average']['RMSE']
    
    # 创建 DataFrame
    df_new = pd.DataFrame([row_data])
    
    # 如果文件存在，追加；否则创建新文件
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"✅ 结果已保存到: {output_path}")

def load_data(csv_path="/ssd1/zhanghongbo04/002/project/NIR-Corn/data/combined_dataset.csv", test_size=0.2, random_state=42):
    """
    加载并预处理数据
    
    Returns:
    - X_train, X_test, y_train, y_test: 划分后的数据
    - scaler_X, scaler_y: 归一化器
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    print(f"📊 加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 提取特征和标签
    X = df.iloc[:, 1:701].values  # Wave_1 ~ Wave_700
    y = df.iloc[:, 701:].values   # Moisture, Starch, Oil, Protein
    
    print(f"✅ 数据形状: X={X.shape}, y={y.shape}")
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"📊 训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            scaler_X, scaler_y, y_train, y_test)

if __name__ == "__main__":
    print("✅ utils.py 已加载")
    print("可用函数:")
    print("  - evaluate_model()")
    print("  - plot_predictions()")
    print("  - save_results_to_csv()")
    print("  - load_data()")