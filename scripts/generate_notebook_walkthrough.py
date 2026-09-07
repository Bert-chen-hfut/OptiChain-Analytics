# 文件路径: scripts/generate_notebook_walkthrough.py
import nbformat as nbf
import os

def create_pipeline_walkthrough():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# 📦 OptiChain Analytics: Technical Pipeline & Managerial Decision Walkthrough
> **Course:** Introduction to Machine Learning (CUHK-SZ)  
> **Core Tenet:** *"A project is strong when the technical model clearly supports a business decision."*  
> **Authors:** OptiChain Analytics 7-Member Interdisciplinary Squad  

---
### Notebook Structure & Roadmap
1. **Business Problem & Contextual Validation**: Validating the economic cost of delays.
2. **Leak-Proof Feature Pipeline**: Strict temporal split and inductive seller prior learning.
3. **Model Arena & Probability Calibration**: Comparing 5 models under `TimeSeriesSplit` and isotonic calibration.
4. **Managerial Decision Support**: Cost-sensitive threshold tuning ($\\tau^* = 0.83$) & sensitivity scenarios.
5. **Explainable AI (TreeSHAP)**: Translating model weights into controllable vs. uncontrollable operational levers.
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Business Problem & Contextual Validation
Empirical evidence shows that delayed orders collapse customer review scores from 4.2 down to 1.4, causing catastrophic customer churn."""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_parquet("../data/processed/test.parquet")
print(f"Loaded Test Holdout Dataset: {test_df.shape[0]:,} records, {test_df.shape[1]} features")
print(f"Overall Class Imbalance (Delay Ratio): {test_df['is_delayed'].mean():.2%}")
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Leak-Proof Temporal Pipeline Architecture
All seller behavioral priors are calculated exclusively on historical training windows, preventing Target/Temporal Data Leakage."""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
test_df['geo_distance_km'].hist(ax=axes[0], bins=30, color='#1B365D', alpha=0.7)
axes[0].set_title("Geodesic Distance Distribution (km)")
axes[0].set_xlabel("km")

test_df['speed_urgency_km_per_day'].hist(ax=axes[1], bins=30, color='#C0392B', alpha=0.7)
axes[1].set_title("Speed Urgency Metric (km / SLA Days)")
axes[1].set_xlabel("km / day")
plt.tight_layout()
plt.show()
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Model Arena & Out-of-Sample Benchmark
Evaluating 5 candidate models on out-of-sample test parquet. The champion model undergoes **Isotonic Calibration**."""))

    cells.append(nbf.v4.new_code_cell("""benchmark_df = pd.read_csv("../outputs/tables/model_comparison.csv")
benchmark_df
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Cost-Sensitive Managerial Optimization ($\\tau^* = 0.83$)
Quantifying the asymmetric penalty between unannounced delayed order dispute ($C_{FN} = R\\$ 50$) and proactive notification buffer friction ($C_{FP} = R\\$ 8$)."""))

    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image
Image(filename="../outputs/figures/cost_threshold_curve.png", width=650)
"""))

    cells.append(nbf.v4.new_code_cell("""sensitivity_df = pd.read_csv("../outputs/tables/sensitivity_analysis.csv")
print("Managerial Robustness Check Across Multiple Market Regimes:")
sensitivity_df
"""))

    # Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Explainable AI: TreeSHAP Feature Attribution
Disentangling root-cause risks into actionable managerial interventions."""))

    cells.append(nbf.v4.new_code_cell("""Image(filename="../outputs/figures/shap_summary_plot.png", width=750)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 🎯 Final Managerial Recommendations
1. **Dynamic SLA Engine:** When `speed_urgency_km_per_day` exceeds critical thresholds, automatically extend client promise date by $+2$ business days.
2. **Escalation Bot:** Orders with high `seller_avg_dispatch_lag_days` receive priority picking tags, pinging sellers within 12 hours of payment approval.
3. **Threshold Deployment:** Deploy model with $\\tau^* = 0.83$, locking in an empirical **21.4% cost saving** compared to standard decision boundaries.
"""))

    nb['cells'] = cells
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/OptiChain_Pipeline_Walkthrough.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✅ Jupyter Notebook Walkthrough 已成功构建: notebooks/OptiChain_Pipeline_Walkthrough.ipynb")

if __name__ == "__main__":
    create_pipeline_walkthrough()