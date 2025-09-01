# scripts/create_combined_dataset.py

import numpy as np
import pandas as pd
import os

def load_processed_data(data_dir="data/processed_interpolated"):
    """加载处理后的数据"""
    print(f"🔄 加载处理后的数据从: {data_dir}")
    
    # 加载原始数据（未归一化）
    X_real = np.load(os.path.join(data_dir, "X_real_original.npy"))      # (80, 700)
    y_real = np.load(os.path.join(data_dir, "y_real_original.npy"))      # (80, 4)
    
    # 加载插值数据（未归一化）
    X_fake = np.load(os.path.join(data_dir, "X_interpolated_original.npy"))  # (920, 700)
    y_fake = np.load(os.path.join(data_dir, "y_interpolated_original.npy"))  # (920, 4)
    
    print(f"✅ 原始数据: 光谱 {X_real.shape}, 标签 {y_real.shape}")
    print(f"✅ 插值数据: 光谱 {X_fake.shape}, 标签 {y_fake.shape}")
    
    return X_real, y_real, X_fake, y_fake

def create_combined_csv(X_real, y_real, X_fake, y_fake, output_path="data/combined_dataset.csv"):
    """创建合并的 CSV 文件"""
    print("🔄 创建合并的 CSV 文件...")
    
    # 合并数据
    X_combined = np.vstack([X_real, X_fake])  # (1000, 700)
    y_combined = np.vstack([y_real, y_fake])  # (1000, 4)
    
    print(f"📊 合并后数据形状: 光谱 {X_combined.shape}, 标签 {y_combined.shape}")
    
    # 创建波长列名
    wavelength_cols = [f'Wave_{i+1}' for i in range(X_combined.shape[1])]
    
    # 创建 DataFrame
    df_spectra = pd.DataFrame(X_combined, columns=wavelength_cols)
    
    # 添加样本 ID
    sample_ids = [f"S{i+1:04d}" for i in range(X_combined.shape[0])]
    df_spectra.insert(0, 'SampleID', sample_ids)
    
    # 添加标签
    target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    df_labels = pd.DataFrame(y_combined, columns=target_names)
    
    # 合并所有列
    df_combined = pd.concat([df_spectra, df_labels], axis=1)
    
    # 保存为 CSV
    df_combined.to_csv(output_path, index=False)
    print(f"✅ 合并 CSV 已保存: {output_path}")
    print(f"📁 文件大小: {df_combined.shape}")
    
    # 显示前几行预览
    print("\n📋 数据预览:")
    print(df_combined.head(3).to_string())
    
    return df_combined

def create_dataset_info(df_combined, output_dir="data"):
    """创建数据集信息文件"""
    info = {
        "total_samples": len(df_combined),
        "spectral_dimensions": 700,
        "target_variables": ['Moisture', 'Starch', 'Oil', 'Protein'],
        "data_source": "Real + Interpolated samples",
        "real_samples": 80,
        "interpolated_samples": 1920
    }
    
    info_path = os.path.join(output_dir, "dataset_info.txt")
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📝 数据集信息已保存: {info_path}")
    
    # 打印统计信息
    print("\n📈 数据集统计:")
    for target in ['Moisture', 'Starch', 'Oil', 'Protein']:
        values = df_combined[target]
        print(f"  {target}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

def main():
    # 加载数据
    X_real, y_real, X_fake, y_fake = load_processed_data()
    
    # 创建合并的 CSV
    df_combined = create_combined_csv(X_real, y_real, X_fake, y_fake)
    
    # 创建数据集信息
    create_dataset_info(df_combined)
    
    print("\n🎉 合并数据集创建完成！")
    print("📁 输出文件:")
    print("  - data/combined_dataset.csv")
    print("  - data/dataset_info.txt")

if __name__ == "__main__":
    main()