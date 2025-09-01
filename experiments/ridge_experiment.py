# experiments/ridge_experiment.py

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import os

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

def run_ridge_experiment():
    """运行 Ridge 回归实验"""
    print("🚀 开始 Ridge 回归实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    # 超参搜索
    print("🔍 进行超参搜索...")
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train_scaled)
    
    # 使用最佳模型
    best_ridge = grid_search.best_estimator_
    print(f"✅ 最佳参数: alpha={grid_search.best_params_['alpha']}")
    
    # 预测
    print("🔮 进行预测...")
    y_pred_scaled = best_ridge.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    
    # 评估
    metrics = evaluate_model(
        y_test_orig, y_pred_orig, 
        model_name="Ridge",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="Ridge",
        save_path="experiments/results/ridge_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "Ridge")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(best_ridge, os.path.join(model_dir, "ridge_model.pkl"))
    joblib.dump(scaler_X, os.path.join(model_dir, "ridge_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "ridge_scaler_y.pkl"))
    
    print("✅ Ridge 回归实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_ridge_experiment()