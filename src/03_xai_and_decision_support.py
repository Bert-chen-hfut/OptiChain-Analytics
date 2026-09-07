import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

def compute_cost(y_true, y_prob, threshold, c_fn=50.0, c_fp=8.0):
    """
    业务不对称损失矩阵计算:
    FN (漏报延误): 导致客诉退款与纠纷仲裁，单笔损失 50 BRL
    FP (误报延误): 前台拉长预期导致潜在转化轻微折损，单笔损失 8 BRL
    """
    y_pred = (y_prob >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    total_cost = (c_fn * fn) + (c_fp * fp)
    return total_cost, fn, fp

def run_xai_and_threshold_optimization():
    print("=" * 80)
    print(">>> [Phase 3] 启动 TreeSHAP 归因解释与非对称商业成本阈值寻优...")
    print("=" * 80)

    # 1. 加载模型与测试集
    model = joblib.load("models/best_model_xgboost.pkl")
    test_df = pd.read_parquet("data/processed/test.parquet")
    X_test = test_df.drop(columns=['is_delayed'])
    y_test = test_df['is_delayed']

    os.makedirs("outputs/figures", exist_ok=True)

    # 2. TreeSHAP 归因分析 (采样 1500 条样本以兼顾速度与统计代表性)
    print("--> 1/2 正在计算 TreeSHAP 归因解释图...")
    sample_X = X_test.sample(1500, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample_X)

    # (a) 全局特征重要性 Summary Plot (PPT Slide 6 必备)
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values, sample_X, max_display=10, show=False)
    plt.title("TreeSHAP Global Feature Importance (Top 10 Drivers)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary_plot.png")
    plt.close()

    # (b) 单样本局部瀑布图 (用于展示高危订单个案诊断)
    plt.figure(figsize=(10, 6), dpi=300)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_waterfall_example.png")
    plt.close()
    print("✅ SHAP 解释图已成功生成至 outputs/figures/")

    # 3. 商业期望损失与最优分类阈值寻优 (Cost-Sensitive Threshold Tuning)
    print("--> 2/2 正在基于商业代价矩阵 (C_FN=50, C_FP=8) 寻找最优决策阈值...")
    preds = pd.read_csv("outputs/tables/test_predictions.csv")
    y_true = preds['y_true'].values
    y_prob = preds['y_prob_xgb'].values

    thresholds = np.linspace(0.05, 0.95, 91)
    costs, fns, fps = [], [], []

    for t in thresholds:
        c, fn, fp = compute_cost(y_true, y_prob, threshold=t)
        costs.append(c)
        fns.append(fn)
        fps.append(fp)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    default_cost, d_fn, d_fp = compute_cost(y_true, y_prob, threshold=0.5)
    cost_saving_pct = (default_cost - min_cost) / default_cost * 100

    print("-" * 80)
    print(f"【财务量化结论】")
    print(f"  • 默认阈值 (0.50) 总期望损失: {default_cost:,.0f} BRL (漏报 FN={d_fn}, 误报 FP={d_fp})")
    print(f"  • 商业最优阈值 (tau* = {best_threshold:.2f}) 期望损失: {min_cost:,.0f} BRL")
    print(f"  • 净减少平台财务损失: {cost_saving_pct:.2f}% (单月样本外测试集省下 {(default_cost - min_cost):,.0f} BRL)")
    print("-" * 80)

    # 绘制成本-阈值折线图 (供 PPT Slide 5/8 汇报使用)
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot(thresholds, costs, color='#1f77b4', lw=2.5, label='Total Business Expected Cost (BRL)')
    plt.axvline(0.5, color='gray', linestyle='--', label='Default Threshold (0.50)')
    plt.axvline(best_threshold, color='crimson', linestyle='-', lw=2, label=f'Optimal Threshold tau*={best_threshold:.2f}')
    plt.scatter([best_threshold], [min_cost], color='crimson', s=90, zorder=5)
    plt.title("Expected Business Cost vs. Classification Threshold", fontsize=12)
    plt.xlabel("Classification Decision Threshold", fontsize=10)
    plt.ylabel("Total Expected Cost (BRL)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/cost_threshold_curve.png")
    plt.close()
    print("✅ 成本-阈值曲线已导出至 outputs/figures/cost_threshold_curve.png")

if __name__ == "__main__":
    run_xai_and_threshold_optimization()
