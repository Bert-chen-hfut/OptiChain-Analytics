# src/02_model_training.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def evaluate_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'Recall (Delay)': round(recall_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4),
        'PR-AUC': round(average_precision_score(y_true, y_prob), 4)
    }

def run_model_training_and_calibration():
    print("=" * 80)
    print(">>> [Phase 2: 模型竞技场 + TimeSeriesSplit + 保序概率校准] 启动...")
    print("=" * 80)

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train, y_train = train_df.drop(columns=['is_delayed']), train_df['is_delayed']
    X_test, y_test = test_df.drop(columns=['is_delayed']), test_df['is_delayed']
    X_test = X_test[X_train.columns]

    neg_pos_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    results = []

    # 1. 业务基准 (Heuristic Rule)
    rule_pred = ((X_test['geo_distance_km'] > 1200) & (X_test['is_interstate'] == 1)).astype(int)
    rule_prob = (X_test['geo_distance_km'] / X_test['geo_distance_km'].max()).values
    res_rule = evaluate_metrics(y_test, rule_pred, rule_prob)
    res_rule['Model'] = 'Heuristic Baseline (Rule)'
    results.append(res_rule)

    # 2. 候选模型
    models = {
        'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, scale_pos_weight=neg_pos_ratio, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost (Raw)': XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=neg_pos_ratio, learning_rate=0.05, eval_metric='logloss', random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"--> 正在训练模型: {name} ...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        res = evaluate_metrics(y_test, y_pred, y_prob)
        res['Model'] = name
        results.append(res)
        trained_models[name] = model

    # 3. 核心强化：对表现最优的 XGBoost 进行 TimeSeriesSplit + Isotonic 校准
    print("--> 正在执行 XGBoost 的时序交叉验证保序概率校准 (Isotonic Calibration)...")
    tscv = TimeSeriesSplit(n_splits=3)
    calibrated_xgb = CalibratedClassifierCV(
        estimator=models['XGBoost (Raw)'], method='isotonic', cv=tscv
    )
    calibrated_xgb.fit(X_train, y_train)
    cal_prob = calibrated_xgb.predict_proba(X_test)[:, 1]
    cal_pred = (cal_prob >= 0.5).astype(int)
    res_cal = evaluate_metrics(y_test, cal_pred, cal_prob)
    res_cal['Model'] = 'XGBoost (Calibrated, Champion)'
    results.append(res_cal)

    results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Balanced Accuracy', 'Recall (Delay)', 'Precision', 'ROC-AUC', 'PR-AUC']]
    
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
    results_df.to_markdown("outputs/tables/model_comparison.md", index=False)
    
    joblib.dump(calibrated_xgb, "models/best_model_calibrated_xgb.pkl")
    pd.DataFrame({'y_true': y_test.values, 'y_prob': cal_prob}).to_csv("outputs/tables/test_predictions.csv", index=False)

    print("\n🏆 全模型横向评测完成，对比大表已导出！")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_model_training_and_calibration()