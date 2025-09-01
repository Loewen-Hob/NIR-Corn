# experiments/svr_experiment.py

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# 导入工具函数
from utils import load_data, evaluate_model, plot_predictions, save_results_to_csv

def run_svr_experiment():
    """运行 SVR 实验"""
    print("🚀 开始 SVR 实验...")
    
    # 加载数据
    (X_train, X_test, y_train_scaled, y_test_scaled, 
     scaler_X, scaler_y, y_train_orig, y_test_orig) = load_data()
    
    # 由于 SVR 不支持多输出，使用 MultiOutputRegressor
    print("🔍 进行超参搜索...")
    svr = SVR()
    multi_svr = MultiOutputRegressor(svr)
    
    # 简化参数搜索以节省时间
    param_grid = {
        'estimator__C': [1, 10, 100],
        'estimator__gamma': ['scale', 'auto', 0.001, 0.01],
        'estimator__kernel': ['rbf']
    }
    
    grid_search = GridSearchCV(
        multi_svr, param_grid, cv=3, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    # 只用部分数据进行搜索（节省时间）
    search_indices = np.random.choice(len(X_train), min(200, len(X_train)), replace=False)
    X_search = X_train[search_indices]
    y_search = y_train_scaled[search_indices]
    
    grid_search.fit(X_search, y_search)
    
    # 使用最佳模型
    best_svr = grid_search.best_estimator_
    print(f"✅ 最佳参数: {grid_search.best_params_}")
    
    # 预测
    print("🔮 进行预测...")
    y_pred_scaled = best_svr.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
    
    # 评估
    metrics = evaluate_model(
        y_test_orig, y_pred_orig, 
        model_name="SVR",
        scaler_y=None
    )
    
    # 画图
    plot_predictions(
        y_test_orig, y_pred_orig,
        model_name="SVR",
        save_path="experiments/results/svr_predictions.png"
    )
    
    # 保存结果到 CSV
    save_results_to_csv(metrics, "SVR")
    
    # 保存模型和归一化器
    model_dir = "experiments/models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(best_svr, os.path.join(model_dir, "svr_model.pkl"))
    joblib.dump(scaler_X, os.path.join(model_dir, "svr_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(model_dir, "svr_scaler_y.pkl"))
    
    print("✅ SVR 实验完成")
    print(f"💾 模型已保存到: {model_dir}")
    
    return metrics

if __name__ == "__main__":
    run_svr_experiment()