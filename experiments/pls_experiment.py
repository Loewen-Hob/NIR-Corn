
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import joblib
import os

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

def run_pls_experiment():
    """运行 PLS 实验"""
    print("🚀 开始 PLS 实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    # 训练 PLS 模型
    print("🧠 训练 PLS 模型...")
    pls = PLSRegression(n_components=10)
    pls.fit(X_train, y_train_scaled)
    
    # 预测
    print("🔮 进行预测...")
    y_pred_scaled = pls.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    
    # 评估
    metrics = evaluate_model(
        y_test_orig, y_pred_orig, 
        model_name="PLS",
        scaler_y=None  # 已经还原了单位
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="PLS",
        save_path="experiments/results/pls_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "PLS")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(pls, os.path.join(model_dir, "pls_model.pkl"))
    joblib.dump(scaler_X, os.path.join(model_dir, "pls_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "pls_scaler_y.pkl"))
    
    print("✅ PLS 实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_pls_experiment()