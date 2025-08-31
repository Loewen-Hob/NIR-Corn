# scripts/preprocess.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib  # 用于保存 scaler

def main():
    # -------------------------------
    # 1. 路径设置
    # -------------------------------
    input_csv = "/Users/zhanghongbo04/Downloads/video/zhb_test/foods/data/raw/corn_mp5_regression_data.csv"
    output_dir = "/Users/zhanghongbo04/Downloads/video/zhb_test/foods/data/processed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # 2. 读取数据
    # -------------------------------
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 光谱列：Wave_1 到 Wave_700
    spectrum_cols = [col for col in df.columns if col.startswith("Wave_")]
    X = df[spectrum_cols].values  # (80, 700)
    
    # 属性列
    target_names = ['Moisture', 'Starch', 'Oil', 'Protein']
    y = df[target_names].values   # (80, 4)
    
    print(f"Original spectra shape: {X.shape}")
    print(f"Original labels shape: {y.shape}")

    # -------------------------------
    # 3. 数据归一化
    # -------------------------------

    # --- 光谱：使用 StandardScaler（Z-score: 均值0，方差1）---
    # 适合光谱，保留分布特性
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)  # (80, 700)

    # --- 标签：使用 MinMaxScaler（缩放到 [0,1] 或 [-1,1]）---
    # 更适合属性预测，物理范围清晰
    y_scaler = MinMaxScaler(feature_range=(0, 1))  # 也可以用 (-1, 1)
    y_scaled = y_scaler.fit_transform(y)  # (80, 4)

    print("✅ Data normalized:")
    print(f"  X: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f} (after StandardScaler)")
    print(f"  y: min={y_scaled.min():.3f}, max={y_scaled.max():.3f} (after MinMaxScaler)")

    # -------------------------------
    # 4. 保存归一化后的数据
    # -------------------------------
    np.save(os.path.join(output_dir, "train_spectra.npy"), X_scaled)
    np.save(os.path.join(output_dir, "train_labels.npy"), y_scaled)
    
    # -------------------------------
    # 5. 保存 Scaler 对象（关键！用于后续逆变换）
    # -------------------------------
    joblib.dump(X_scaler, os.path.join(output_dir, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(output_dir, "y_scaler.pkl"))
    
    print(f"✅ Saved normalized data and scalers to {output_dir}")

    # -------------------------------
    # 6. 可选：打印原始 vs 归一化范围（用于检查）
    # -------------------------------
    print("\n📌 Scaler info (for reference):")
    print("Moisture range:", y[:, 0].min(), "→", y[:, 0].max())
    print("Starch range:", y[:, 1].min(), "→", y[:, 1].max())
    print("Oil range:", y[:, 2].min(), "→", y[:, 2].max())
    print("Protein range:", y[:, 3].min(), "→", y[:, 3].max())


if __name__ == "__main__":
    main()