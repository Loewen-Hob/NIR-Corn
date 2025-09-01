# scripts/generate_interpolated_data.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path):
    """加载并预处理原始数据"""
    print(f"📊 加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 提取光谱和标签
    spectrum_cols = [col for col in df.columns if col.startswith("Wave_")]
    X = df[spectrum_cols].values  # (80, 700)
    
    target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    y = df[target_names].values   # (80, 4)
    
    print(f"✅ 原始数据形状: 光谱 {X.shape}, 标签 {y.shape}")
    return X, y, spectrum_cols, target_names

def generate_interpolated_samples(X_real, y_real, num_samples=920, noise_level=0.0):
    """
    生成插值样本
    
    Parameters:
    - X_real: 原始光谱 (80, 700)
    - y_real: 原始标签 (80, 4)
    - num_samples: 要生成的样本数
    - noise_level: 可选噪声水平 (0.0 = 无噪声)
    """
    print(f"🔄 开始生成 {num_samples} 个插值样本...")
    
    n_real = X_real.shape[0]
    X_interpolated = []
    y_interpolated = []
    
    for i in range(num_samples):
        # 随机选择两个样本
        idx1, idx2 = np.random.choice(n_real, 2, replace=False)
        
        # 随机插值系数
        alpha = np.random.uniform(0.1, 0.9)  # 避免端点
        
        # 插值计算
        interpolated_spectrum = alpha * X_real[idx1] + (1 - alpha) * X_real[idx2]
        interpolated_label = alpha * y_real[idx1] + (1 - alpha) * y_real[idx2]
        
        # 可选：添加小噪声
        if noise_level > 0:
            noise_spec = np.random.normal(0, noise_level, size=interpolated_spectrum.shape)
            interpolated_spectrum += noise_spec
        
        X_interpolated.append(interpolated_spectrum)
        y_interpolated.append(interpolated_label)
    
    X_interpolated = np.array(X_interpolated)
    y_interpolated = np.array(y_interpolated)
    
    print(f"✅ 生成完成: 光谱 {X_interpolated.shape}, 标签 {y_interpolated.shape}")
    return X_interpolated, y_interpolated

def save_processed_data(X_real, y_real, X_fake, y_fake, output_dir="data/processed_interpolated"):
    """保存处理后的数据（包括归一化）"""
    print(f"💾 保存处理后数据到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 归一化
    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 对真实+插值数据一起归一化（保持一致性）
    X_combined = np.vstack([X_real, X_fake])
    y_combined = np.vstack([y_real, y_fake])
    
    X_scaled = X_scaler.fit_transform(X_combined)
    y_scaled = y_scaler.fit_transform(y_combined)
    
    # 分离回真实和插值数据
    n_real = X_real.shape[0]
    X_real_scaled = X_scaled[:n_real]
    X_fake_scaled = X_scaled[n_real:]
    y_real_scaled = y_scaled[:n_real]
    y_fake_scaled = y_scaled[n_real:]
    
    # 保存归一化后的数据
    np.save(os.path.join(output_dir, "X_real_scaled.npy"), X_real_scaled)
    np.save(os.path.join(output_dir, "y_real_scaled.npy"), y_real_scaled)
    np.save(os.path.join(output_dir, "X_interpolated_scaled.npy"), X_fake_scaled)
    np.save(os.path.join(output_dir, "y_interpolated_scaled.npy"), y_fake_scaled)
    
    # 保存原始值（用于评估）
    np.save(os.path.join(output_dir, "X_real_original.npy"), X_real)
    np.save(os.path.join(output_dir, "y_real_original.npy"), y_real)
    np.save(os.path.join(output_dir, "X_interpolated_original.npy"), X_fake)
    np.save(os.path.join(output_dir, "y_interpolated_original.npy"), y_fake)
    
    # 保存 scaler
    joblib.dump(X_scaler, os.path.join(output_dir, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(output_dir, "y_scaler.pkl"))
    
    print("✅ 数据保存完成")
    return X_scaler, y_scaler

def visualize_samples(X_real, X_fake, y_real, y_fake, num_samples=5):
    """可视化真实 vs 插值样本"""
    print("📊 生成对比图...")
    
    # 确保不超过实际样本数
    num_samples = min(num_samples, len(X_real), len(X_fake))
    
    # 光谱对比 (最多显示 5 个)
    plt.figure(figsize=(15, 10))
    
    # 第一行：光谱对比
    for i in range(min(4, num_samples)):  # 最多 4 个光谱对比
        plt.subplot(2, 4, i+1)
        plt.plot(X_real[i], alpha=0.8, label='Real', linewidth=1)
        plt.plot(X_fake[i], '--', alpha=0.8, label='Interpolated', linewidth=1)
        plt.title(f'Spectrum {i+1}')
        plt.ylabel('Intensity')
        if i == 0:
            plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 第二行：标签分布对比
    target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    for i in range(4):
        plt.subplot(2, 4, 4 + i + 1)  # 从第 5 个位置开始
        plt.hist(y_real[:, i], bins=15, alpha=0.7, label='Real', color='blue', density=True)
        plt.hist(y_fake[:, i], bins=30, alpha=0.7, label='Interpolated', color='orange', density=True)
        plt.xlabel(target_names[i], fontsize=8)
        plt.ylabel('Density', fontsize=8)
        plt.title(f'{target_names[i]}', fontsize=10)
        if i == 3:  # 只在最后一个图例显示
            plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=6)
    
    plt.tight_layout()
    output_path = "src/saved_samples/interpolation_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图已保存: {output_path}")
def main():
    # 路径设置
    csv_path = "/ssd1/zhanghongbo04/002/project/NIR-Corn/data/raw/corn_mp5_regression_data.csv"
    output_dir = "data/processed_interpolated"
    
    # 1. 加载原始数据
    X_real, y_real, spectrum_cols, target_names = load_and_preprocess_data(csv_path)
    
    # 2. 生成插值样本
    X_fake, y_fake = generate_interpolated_samples(
        X_real, y_real, 
        num_samples=1920, 
        noise_level=0.01  # 添加轻微噪声使样本更自然
    )
    
    # 3. 保存数据
    X_scaler, y_scaler = save_processed_data(X_real, y_real, X_fake, y_fake, output_dir)
    
    # 4. 可视化对比
    visualize_samples(X_real, X_fake, y_real, y_fake)
    
    # 5. 打印统计信息
    print("\n📈 数据统计:")
    print(f"原始样本数: {X_real.shape[0]}")
    print(f"插值样本数: {X_fake.shape[0]}")
    print(f"总样本数: {X_real.shape[0] + X_fake.shape[0]}")
    
    for i, name in enumerate(target_names):
        real_min, real_max = y_real[:, i].min(), y_real[:, i].max()
        fake_min, fake_max = y_fake[:, i].min(), y_fake[:, i].max()
        print(f"{name}: 真实[{real_min:.2f}, {real_max:.2f}] 生成[{fake_min:.2f}, {fake_max:.2f}]")

if __name__ == "__main__":
    main()