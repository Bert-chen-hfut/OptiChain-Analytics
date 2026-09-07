import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    recall_score, precision_score, 
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'Recall (Delay)': round(recall_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4),
        'PR-AUC': round(average_precision_score(y_true, y_prob), 4)
    }

def run_model_arena():
    print("=" * 80)
    print(">>> [Phase 2] 启动多模型竞技场与样本外盲测泛化评估...")
    print("=" * 80)

    # 1. 加载处理好的因果时序宽表
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=['is_delayed'])
    y_train = train_df['is_delayed']
    X_test = test_df.drop(columns=['is_delayed'])
    y_test = test_df['is_delayed']

    # 保证测试集列顺序与训练集 100% 对齐
    X_test = X_test[X_train.columns]

    # 计算负正样本比例，设置 scale_pos_weight 代替破坏地理特征的 SMOTE
    neg_pos_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"--> 训练集不平衡比例: scale_pos_weight = {neg_pos_ratio:.2f}")

    results = []

    # 2. Heuristic Baseline (业务经验基准: 运距 > 1200km 且跨州即预测延误)
    print("--> 1/5 评估基准规则模型 (Heuristic Rule Baseline)...")
    rule_pred = ((X_test['geo_distance_km'] > 1200) & (X_test['is_interstate'] == 1)).astype(int)
    rule_prob = (X_test['geo_distance_km'] / X_test['geo_distance_km'].max()).values
    res_rule = evaluate_predictions(y_test, rule_pred, rule_prob)
    res_rule['Model'] = 'Heuristic Baseline (Rule)'
    results.append(res_rule)

    # 3. 候选 ML 模型库
    models = {
        'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, scale_pos_weight=neg_pos_ratio, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost (Fine-tuned)': XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=neg_pos_ratio, learning_rate=0.05, eval_metric='logloss', random_state=42)
    }

    trained_models = {}
    for i, (name, model) in enumerate(models.items(), start=2):
        print(f"--> {i}/5 正在训练并盲测模型: {name} ...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics['Model'] = name
        results.append(metrics)
        trained_models[name] = model

    # 4. 生成横向对比大表 (供 Report 组写入第 4 章，供 PPT 组直接制表)
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'Balanced Accuracy', 'Recall (Delay)', 'Precision', 'ROC-AUC', 'PR-AUC']]

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results_df.to_csv("outputs/tables/model_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("🏆 全模型样本外测试集 (Test Set) 核心性能对比大表已生成：")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # 5. 保存最优主力模型与盲测预测概率 (供 Phase 3 SHAP 与成本寻优使用)
    best_model = trained_models['XGBoost (Fine-tuned)']
    joblib.dump(best_model, "models/best_model_xgboost.pkl")

    test_preds_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_prob_xgb': best_model.predict_proba(X_test)[:, 1]
    })
    test_preds_df.to_csv("outputs/tables/test_predictions.csv", index=False)

    print("✅ 最优模型已保存至: models/best_model_xgboost.pkl")
    print("✅ 盲测集预测结果已保存至: outputs/tables/test_predictions.csv")
    print("=" * 80)

if __name__ == "__main__":
    run_model_arena()
