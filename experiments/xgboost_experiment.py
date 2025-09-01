# experiments/xgboost_experiment.py

import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
import time

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

def run_xgboost_experiment():
    """运行 XGBoost 实验"""
    print("🚀 开始 XGBoost 实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    # 为了加快训练速度，使用部分数据进行训练
    print("🔍 训练 XGBoost 模型...")
    
    # 使用更少的 estimators 和更简单的参数
    xgb_params = {
        'random_state': 42,
        'n_estimators': 50,      # 减少树的数量
        'max_depth': 4,          # 减少深度
        'learning_rate': 0.1,
        'subsample': 0.8,        # 随机采样
        'colsample_bytree': 0.8, # 特征采样
        'n_jobs': -1             # 使用所有 CPU 核心
    }
    
    # 创建多输出回归器
    multi_xgb = MultiOutputRegressor(
        XGBRegressor(**xgb_params),
        n_jobs=1  # 避免嵌套并行
    )
    
    # 训练模型（添加进度提示）
    print("⏳ 正在训练模型，请稍候...")
    start_time = time.time()
    
    try:
        multi_xgb.fit(X_train, y_train_scaled)
        train_time = time.time() - start_time
        print(f"✅ 模型训练完成，耗时: {train_time:.2f} 秒")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        # 尝试更简单的参数
        print("🔄 尝试更简单的参数...")
        simple_params = {
            'random_state': 42,
            'n_estimators': 30,
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_jobs': -1
        }
        multi_xgb = MultiOutputRegressor(XGBRegressor(**simple_params), n_jobs=1)
        multi_xgb.fit(X_train, y_train_scaled)
        print("✅ 使用简化参数训练完成")
    
    # 预测
    print("🔮 进行预测...")
    y_pred_scaled = multi_xgb.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    
    # 评估
    metrics = evaluate_model(
        y_test_orig, y_pred_orig, 
        model_name="XGBoost",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="XGBoost",
        save_path="experiments/results/xgboost_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "XGBoost")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(multi_xgb, os.path.join(model_dir, "xgboost_model.pkl"))
    joblib.dump(scaler_X, os.path.join(model_dir, "xgboost_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "xgboost_scaler_y.pkl"))
    
    print("✅ XGBoost 实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_xgboost_experiment()