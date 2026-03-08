"""
模型下载模块
从 configs/model_info/ 目录读取模型列表，支持 mlx / original 两种来源
模型按 框架/Company/Series/ModelName 目录结构存放

独立运行: python -m modules.download_model
"""

import os
import json
import time
from huggingface_hub import snapshot_download

# ============================================================
# 📦 配置
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(ROOT_DIR, "models")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs", "model_info")


# ============================================================
# 🔧 核心函数
# ============================================================

def load_models_config(source="mlx"):
    """
    从 JSON 配置文件加载模型列表

    Returns:
        list[dict]: [{"Company": "...", "Series": "...", "Models": [...]}]
    """
    filename = f"{source}.json"
    config_path = os.path.join(CONFIGS_DIR, filename)
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_companies(source="mlx"):
    """获取所有公司名称"""
    config = load_models_config(source)
    return sorted(set(item["Company"] for item in config))


def get_series(company, source="mlx"):
    """获取指定公司的所有系列"""
    config = load_models_config(source)
    return sorted(set(
        item["Series"] for item in config if item["Company"] == company
    ))


def get_models(company, series, source="mlx"):
    """获取指定公司 + 系列下的模型列表"""
    config = load_models_config(source)
    for item in config:
        if item["Company"] == company and item["Series"] == series:
            return item.get("Models", [])
    return []


def get_framework_inference(company, series, source="mlx"):
    """获取指定公司+系列支持的推理框架列表"""
    from modules.core_types import Framework
    config = load_models_config(source)
    for item in config:
        if item["Company"] == company and item["Series"] == series:
            frameworks = item.get("FrameworkInference")
            if frameworks:
                return [Framework(f) for f in frameworks]
    # 未配置时按来源默认
    if source == "mlx":
        return [Framework.MLX]
    return [Framework.TRANSFORMERS]


def find_model_info(model_name, source="mlx"):
    """根据模型名称反查 Company 和 Series"""
    config = load_models_config(source)
    for item in config:
        if model_name in item.get("Models", []):
            return item["Company"], item["Series"]
    return None, None


def get_repo_id(model_name, source="mlx"):
    """根据模型名称和来源拼接 repo_id"""
    if source == "mlx":
        return f"mlx-community/{model_name}"
    return model_name


def get_local_path(model_name, source="mlx", company=None, series=None):
    """
    获取模型本地路径: models/source/Company/Series/ModelName

    company/series 可选，未提供时自动从配置中查找
    """
    if not company or not series:
        company, series = find_model_info(model_name, source)

    repo_id = get_repo_id(model_name, source)
    local_name = repo_id.split("/")[-1]

    if company and series:
        return os.path.join(LOCAL_DIR, source, company, series, local_name)
    # 兜底：找不到配置时放在 models/source/ 下
    return os.path.join(LOCAL_DIR, source, local_name)


def model_exists(model_name, source="mlx", company=None, series=None):
    """检测本地是否已存在该模型"""
    local_path = get_local_path(model_name, source, company, series)
    return os.path.exists(os.path.join(local_path, "config.json"))


def scan_local_models():
    """
    扫描本地已下载的模型（遍历 source/Company/Series/Model 目录结构）

    Returns:
        list[dict]: [{"source": "...", "company": "...", "series": "...", "model": "...", "path": "...", "label": "..."}]
    """
    if not os.path.exists(LOCAL_DIR):
        return []
    results = []
    for source in sorted(os.listdir(LOCAL_DIR)):
        source_dir = os.path.join(LOCAL_DIR, source)
        if not os.path.isdir(source_dir):
            continue
        for company in sorted(os.listdir(source_dir)):
            company_dir = os.path.join(source_dir, company)
            if not os.path.isdir(company_dir):
                continue
            for series in sorted(os.listdir(company_dir)):
                series_dir = os.path.join(company_dir, series)
                if not os.path.isdir(series_dir):
                    continue
                for model in sorted(os.listdir(series_dir)):
                    model_dir = os.path.join(series_dir, model)
                    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
                        results.append({
                            "source": source,
                            "company": company,
                            "series": series,
                            "model": model,
                            "path": model_dir,
                            "label": f"[{source}] {company} / {series} / {model}",
                        })
    return results


def download(model_name, source="mlx", company=None, series=None):
    """
    下载模型到本地

    Returns:
        (local_path, elapsed) 下载成功
    Raises:
        FileExistsError: 模型已存在
    """
    repo_id = get_repo_id(model_name, source)
    local_path = get_local_path(model_name, source, company, series)

    if os.path.exists(os.path.join(local_path, "config.json")):
        raise FileExistsError(f"模型已存在: {local_path}")

    print(f"⏳ 开始下载 {repo_id}")
    print(f"📂 保存路径: {local_path}")

    os.makedirs(local_path, exist_ok=True)
    s = time.time()
    snapshot_download(repo_id=repo_id, local_dir=local_path)
    elapsed = time.time() - s

    print(f"✅ 下载完成，耗时 {elapsed:.2f} 秒")
    return local_path, elapsed


# ============================================================
# 🚀 独立运行：交互式下载
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("📥 模型下载工具")
    print("=" * 50)

    # 选择来源
    print("\n模型来源:")
    print("  1. mlx（已量化 MLX 格式，推荐 Mac）")
    print("  2. original（原始 HuggingFace 模型）")
    source_input = input("\n请选择 [1/2]（默认 1）: ").strip() or "1"
    source = "mlx" if source_input == "1" else "original"

    # 选择公司
    companies = get_companies(source)
    if not companies:
        print("❌ 未找到模型配置，请检查 configs/ 目录")
        exit(1)

    print(f"\n公司/组织:")
    for i, c in enumerate(companies, 1):
        print(f"  {i}. {c}")
    idx = int(input(f"\n请选择 [1-{len(companies)}]（默认 1）: ").strip() or "1") - 1
    company = companies[idx]

    # 选择系列
    series_list = get_series(company, source)
    print(f"\n模型系列（{company}）:")
    for i, s in enumerate(series_list, 1):
        print(f"  {i}. {s}")
    idx = int(input(f"\n请选择 [1-{len(series_list)}]（默认 1）: ").strip() or "1") - 1
    series = series_list[idx]

    # 选择模型
    models = get_models(company, series, source)
    if not models:
        print("❌ 该系列下暂无模型")
        exit(1)

    print(f"\n可用模型（{company} / {series}）:")
    for i, name in enumerate(models, 1):
        exists = "✅ 已下载" if model_exists(name, source, company, series) else ""
        print(f"  {i}. {name}  {exists}")

    idx = int(input(f"\n请选择模型编号 [1-{len(models)}]（默认 1）: ").strip() or "1") - 1
    model_name = models[idx]

    # 检测是否存在
    if model_exists(model_name, source, company, series):
        print(f"\n✅ 模型已存在本地，无需下载")
        print(f"📂 路径: {get_local_path(model_name, source, company, series)}")
        exit(0)

    # 确认下载
    repo_id = get_repo_id(model_name, source)
    confirm = input(f"\n确认下载 {repo_id}？[y/N]: ").strip().lower()
    if confirm != "y":
        print("已取消")
        exit(0)

    try:
        download(model_name, source, company, series)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        exit(1)
