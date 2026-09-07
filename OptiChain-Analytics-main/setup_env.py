import os
import shutil
import subprocess
import sys

def run_cmd(cmd):
    """运行终端命令并忽略非致命错误"""
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print(f"执行提示: {e}")

def main():
    print("=" * 70)
    print(">>> [Phase 0] 开始初始化 OptiChain 项目架构与仓库净化...")
    print("=" * 70)

    # 1. 创建标准工程目录结构
    dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "outputs/figures",
        "outputs/tables",
        "app",
        "models"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ [1/5] 标准目录创建完成: data/, notebooks/, src/, outputs/, app/, models/")

    # 2. 检查并清理旧项目残留 (ReturnIQ 目录)
    old_dir = "ReturnIQ-AI-Return-Prediction-main"
    if os.path.exists(old_dir):
        old_raw = os.path.join(old_dir, "data", "raw")
        if os.path.exists(old_raw):
            print(f"正在从 {old_dir} 抢救原始 CSV 数据集...")
            for fname in os.listdir(old_raw):
                src_file = os.path.join(old_raw, fname)
                dst_file = os.path.join("data", "raw", fname)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
        
        # 物理删除旧文件夹
        shutil.rmtree(old_dir, ignore_errors=True)
        # 从 Git 缓存中移除追踪
        run_cmd(f"git rm -r --cached {old_dir}")
        print("✅ [2/5] 旧项目残留已彻底物理抹除，并已从 Git 缓存中注销！")
    else:
        print("ℹ️ [2/5] 未检测到旧项目文件夹，跳过清理。")

    # 3. 写入规范的 .gitignore (严防大文件爆库)
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Environments
venv/
env/
.venv/

# Jupyter
.ipynb_checkpoints/

# Large Datasets
data/raw/*.csv
data/raw/*.zip
data/processed/*.parquet
data/processed/*.csv

# Model Weights
models/*.pkl
models/*.joblib

# OS files
.DS_Store
Thumbs.db
"""
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("✅ [3/5] .gitignore 文件配置完成 (已自动屏蔽大文件与权重)。")

    # 4. 生成统一 requirements.txt
    req_content = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.43.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
joblib>=1.3.0
pyarrow>=12.0.0
"""
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(req_content)
    print("✅ [4/5] requirements.txt 文件已更新。")

    # 5. 检查 Olist 数据集是否齐全
    required_files = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_sellers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_payments_dataset.csv"
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join("data", "raw", f))]
    print("\n" + "-" * 70)
    if not missing:
        print("✅ [5/5] 完美！7 张 Olist 核心 CSV 数据集已全部就绪！")
    else:
        print(f"⚠️ [5/5] data/raw/ 目录下还缺少以下 {len(missing)} 个数据文件：")
        for m in missing:
            print(f"   - {m}")
        print("\n【操作指引】请将从 Kaggle 下载的 Olist 数据集解压后的 CSV 放入 data/raw/ 文件夹中。")
    print("-" * 70)

    # 6. 自动同步一次 Git 状态
    print("\n>>> 正在同步 Git 提交...")
    run_cmd("git add .gitignore requirements.txt setup_env.py")
    run_cmd('git commit -m "chore: initialize project structure and clean legacy directory"')
    print("==================================================================")
    print("🎉 Phase 0 初始化完成！")
    print("==================================================================")

if __name__ == "__main__":
    main()