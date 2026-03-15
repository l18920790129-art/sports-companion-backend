"""
RAG（检索增强生成）引擎 v1.0
为运动路线推荐系统提供知识库检索能力

核心功能：
1. 厦门运动路线知识库（包含地标、路面、景观、设施等详细信息）
2. 基于TF-IDF的语义相似度检索
3. 历史查询记录检索（长期记忆）
4. 知识图谱节点关联查询

数据来源：
- 厦门市地理信息公共服务平台
- OpenStreetMap厦门数据集
- 高德地图POI数据
- 人工整理的运动路线经验数据
"""
import os
import json
import math
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ============================================================
# 厦门运动路线知识库（真实地理信息）
# ============================================================
XIAMEN_ROUTE_KNOWLEDGE = [
    # ---- 环岛路海滨线 ----
    {
        "id": "KB001",
        "title": "环岛路海滨跑步线",
        "content": "厦门环岛路是全国最美海滨马拉松赛道之一，全长约43公里，沿厦门岛南部海岸线延伸。"
                   "路面为沥青铺装，平坦宽阔，适合各水平跑者。沿途设有多个补给站和饮水点，"
                   "白城沙滩段（纬度24.44附近）视野开阔，可远眺鼓浪屿。"
                   "椰风寨至曾厝垵段约4.7公里，是最受欢迎的晨跑路段，"
                   "全程树荫较少（约22%），建议清晨或傍晚运动，避免正午暴晒。",
        "tags": ["海景", "平坦", "沥青路面", "环岛路", "白城", "椰风寨", "曾厝垵", "马拉松"],
        "route_id": "ROUTE_A",
        "area": "南部海滨",
        "difficulty": "低",
        "shade_pct": 22,
        "soft_surface_pct": 18,
        "water_stations": 3,
        "best_time": "清晨6-8点，傍晚17-19点",
        "lat_range": [24.43, 24.46],
        "lon_range": [118.07, 118.12],
    },
    {
        "id": "KB002",
        "title": "白城沙滩至曾厝垵路线详情",
        "content": "白城沙滩（坐标约24.4402°N, 118.0842°E）是厦门大学南门外的标志性沙滩，"
                   "沙质细腻，是厦门最受欢迎的城市沙滩之一。从白城沙滩出发，"
                   "沿环岛路向东行约4.7公里可到达曾厝垵文创村。"
                   "曾厝垵（坐标约24.4460°N, 118.1115°E）是厦门著名的文艺渔村，"
                   "有大量特色小吃和文创店铺，是运动后补给的好去处。"
                   "该路段地势平缓，海风习习，脚踝友好，适合轻松跑和散步。",
        "tags": ["白城沙滩", "曾厝垵", "文创村", "海景", "脚踝友好", "平坦"],
        "route_id": "ROUTE_A",
        "area": "南部海滨",
        "difficulty": "低",
        "shade_pct": 22,
        "soft_surface_pct": 18,
        "water_stations": 3,
        "best_time": "全天均可",
        "lat_range": [24.43, 24.46],
        "lon_range": [118.07, 118.12],
    },
    # ---- 厦大绿化线 ----
    {
        "id": "KB003",
        "title": "厦大-南普陀寺绿化跑步线",
        "content": "厦门大学校园内设有完善的跑步路线，林荫覆盖率高达65%以上，"
                   "是夏季运动的首选。厦大芙蓉路两侧种植了大量凤凰木和榕树，"
                   "形成天然绿色隧道。南普陀寺（坐标约24.4581°N, 118.0665°E）"
                   "是厦门著名佛教圣地，寺院周边绿化良好，坡度适中。"
                   "从椰风寨出发经厦大校园至南普陀寺约3.5公里，"
                   "路面为软硬混合（校园步道+山路），软路面比例约42%，"
                   "适合有脚踝或膝盖不适的跑者。沿途有多个饮水点。",
        "tags": ["厦大", "树荫", "绿化", "南普陀寺", "软路面", "脚踝友好", "校园"],
        "route_id": "ROUTE_B",
        "area": "中部校园",
        "difficulty": "中",
        "shade_pct": 65,
        "soft_surface_pct": 42,
        "water_stations": 2,
        "best_time": "全天均可，夏季尤佳",
        "lat_range": [24.44, 24.47],
        "lon_range": [118.06, 118.09],
    },
    {
        "id": "KB004",
        "title": "万石山植物园跑步线",
        "content": "万石山植物园（坐标约24.4587°N, 118.0718°E）是厦门最大的城市植物园，"
                   "占地约4.5平方公里，NDVI植被指数均值0.78，是厦门绿化最好的区域之一。"
                   "园内设有多条跑步步道，路面为土路和橡胶跑道，软路面比例高，"
                   "对膝盖和脚踝的冲击较小。园内有多个饮水站，"
                   "全程树荫覆盖率约82%，即使夏季正午也较为凉爽。"
                   "需注意部分路段有坡度，累计爬升约95米，建议中级以上跑者选择。",
        "tags": ["植物园", "万石山", "树荫", "软路面", "爬坡", "绿化", "NDVI高"],
        "route_id": "ROUTE_B",
        "area": "中部山地",
        "difficulty": "中高",
        "shade_pct": 82,
        "soft_surface_pct": 65,
        "water_stations": 2,
        "best_time": "清晨或傍晚",
        "lat_range": [24.45, 24.48],
        "lon_range": [118.07, 118.10],
    },
    # ---- 五缘湾综合线 ----
    {
        "id": "KB005",
        "title": "五缘湾湿地公园跑步线",
        "content": "五缘湾湿地公园（坐标约24.4613°N, 118.0965°E）位于厦门岛北部，"
                   "是厦门岛内最大的湿地公园，占地约390公顷。"
                   "公园内设有完善的跑步步道，路面平整，以橡胶跑道和木栈道为主，"
                   "软路面比例约28%，适合中等强度训练。"
                   "五缘湾内湖景观优美，可观赏白鹭等水鸟，"
                   "沿湾跑步可欣赏帆船和游艇，是厦门最具特色的运动场所之一。"
                   "公园内设有多个补给站，交通便利，有多路公交直达。",
        "tags": ["五缘湾", "湿地公园", "北部", "湖景", "橡胶跑道", "帆船", "白鹭"],
        "route_id": "ROUTE_C",
        "area": "北部湿地",
        "difficulty": "低中",
        "shade_pct": 35,
        "soft_surface_pct": 28,
        "water_stations": 2,
        "best_time": "清晨观鸟最佳",
        "lat_range": [24.46, 24.49],
        "lon_range": [118.08, 118.11],
    },
    {
        "id": "KB006",
        "title": "厦门岛北部莲前路-五缘湾路线",
        "content": "莲前路是厦门岛东西向主干道，连接岛内各区。"
                   "从莲前路北段出发，经五缘湾大桥可到达五缘湾湿地公园。"
                   "该路线地势较平坦，适合骑行和快走。"
                   "五缘湾片区（纬度24.46-24.50）是厦门近年来重点开发的新区，"
                   "配套设施完善，有多个运动场馆和健身步道。"
                   "该区域路面宽阔，车流量相对较少，适合长距离训练。",
        "tags": ["莲前路", "五缘湾", "北部", "骑行", "长距离", "新区"],
        "route_id": "ROUTE_C",
        "area": "北部新区",
        "difficulty": "低",
        "shade_pct": 30,
        "soft_surface_pct": 25,
        "water_stations": 3,
        "best_time": "全天均可",
        "lat_range": [24.46, 24.52],
        "lon_range": [118.07, 118.12],
    },
    # ---- 北部路线（新增）----
    {
        "id": "KB007",
        "title": "厦门岛北部环岛路-五缘湾北线",
        "content": "厦门岛北部区域（纬度24.49-24.54）包括翔安大桥头、五缘湾北岸、"
                   "同集路等区域，是厦门岛内较少被开发的运动区域。"
                   "五缘湾北岸（坐标约24.505°N, 118.095°E）有完整的滨水步道，"
                   "全长约3.2公里，路面为彩色橡胶跑道，软路面比例约55%，"
                   "非常适合有关节不适的跑者。沿途可欣赏五缘湾全景，"
                   "清晨可见帆船训练，景色优美。该路段车流量少，安全性高。",
        "tags": ["北部", "五缘湾北岸", "橡胶跑道", "软路面", "滨水", "安静"],
        "route_id": "ROUTE_D",
        "area": "北部滨水",
        "difficulty": "低",
        "shade_pct": 30,
        "soft_surface_pct": 55,
        "water_stations": 2,
        "best_time": "清晨最佳",
        "lat_range": [24.49, 24.54],
        "lon_range": [118.08, 118.12],
    },
    {
        "id": "KB008",
        "title": "厦门岛北部集美大桥-翔安隧道口路线",
        "content": "厦门岛北部集美大桥头区域（坐标约24.535°N, 118.083°E）"
                   "是厦门岛与集美区的连接点，附近有完善的滨海步道。"
                   "该区域地势平坦，视野开阔，可远眺集美学村和嘉庚建筑群。"
                   "沿海岸线向东可到达翔安隧道口，全程约5公里，"
                   "路面以沥青为主，部分路段有橡胶步道。"
                   "该路线人流量较少，适合追求安静运动环境的跑者。",
        "tags": ["北部", "集美大桥", "翔安", "海景", "安静", "平坦"],
        "route_id": "ROUTE_D",
        "area": "北部海岸",
        "difficulty": "低",
        "shade_pct": 20,
        "soft_surface_pct": 30,
        "water_stations": 1,
        "best_time": "清晨或傍晚",
        "lat_range": [24.50, 24.55],
        "lon_range": [118.07, 118.11],
    },
    # ---- 健康与运动建议 ----
    {
        "id": "KB009",
        "title": "脚踝不适时的运动建议",
        "content": "脚踝不适时，应优先选择软路面路线，避免长时间在硬质水泥路或沥青路面上运动。"
                   "推荐路线：万石山植物园（软路面65%）、五缘湾北岸橡胶跑道（软路面55%）、"
                   "厦大校园步道（软路面42%）。应避免有坡度的路段，减少踝关节侧向受力。"
                   "建议穿着有良好缓震支撑的跑鞋，运动前充分热身，"
                   "运动后进行冰敷处理。运动强度应降低至轻松或中等水平。",
        "tags": ["脚踝", "软路面", "健康约束", "运动建议", "康复"],
        "route_id": None,
        "area": "健康建议",
        "difficulty": None,
        "shade_pct": None,
        "soft_surface_pct": None,
        "water_stations": None,
        "best_time": None,
        "lat_range": None,
        "lon_range": None,
    },
    {
        "id": "KB010",
        "title": "厦门夏季运动注意事项",
        "content": "厦门夏季（6-9月）气温高，湿度大，运动时需特别注意防暑降温。"
                   "建议选择树荫覆盖率高的路线，如万石山植物园（82%）、厦大校园（68%）。"
                   "应避免10:00-16:00时段在环岛路等暴露路段运动。"
                   "沿途水站分布：环岛路约每1.5公里一个，厦大校园约每0.8公里一个。"
                   "建议携带运动饮料，补充电解质，预防中暑。",
        "tags": ["夏季", "防暑", "水站", "树荫", "运动建议"],
        "route_id": None,
        "area": "季节建议",
        "difficulty": None,
        "shade_pct": None,
        "soft_surface_pct": None,
        "water_stations": None,
        "best_time": None,
        "lat_range": None,
        "lon_range": None,
    },
]

# ============================================================
# 知识图谱节点（地点-路线-特征关系）
# ============================================================
KNOWLEDGE_GRAPH = {
    "nodes": {
        # 地点节点
        "白城沙滩":     {"type": "location", "lat": 24.4402, "lon": 118.0842, "area": "南部"},
        "曾厝垵":       {"type": "location", "lat": 24.4460, "lon": 118.1115, "area": "南部"},
        "椰风寨":       {"type": "location", "lat": 24.4383, "lon": 118.0854, "area": "南部"},
        "胡里山炮台":   {"type": "location", "lat": 24.4378, "lon": 118.1185, "area": "南部"},
        "南普陀寺":     {"type": "location", "lat": 24.4581, "lon": 118.0665, "area": "中部"},
        "厦门大学":     {"type": "location", "lat": 24.4508, "lon": 118.0725, "area": "中部"},
        "万石山植物园": {"type": "location", "lat": 24.4587, "lon": 118.0718, "area": "中部"},
        "五缘湾湿地公园": {"type": "location", "lat": 24.4613, "lon": 118.0965, "area": "北部"},
        "五缘湾北岸":   {"type": "location", "lat": 24.505,  "lon": 118.095,  "area": "北部"},
        "集美大桥头":   {"type": "location", "lat": 24.535,  "lon": 118.083,  "area": "北部"},
        "莲前路":       {"type": "location", "lat": 24.490,  "lon": 118.090,  "area": "北部"},
        # 特征节点
        "海景":         {"type": "feature", "category": "scenery"},
        "树荫":         {"type": "feature", "category": "environment"},
        "软路面":       {"type": "feature", "category": "surface"},
        "水站":         {"type": "feature", "category": "facility"},
        "公园":         {"type": "feature", "category": "environment"},
        "平坦":         {"type": "feature", "category": "terrain"},
        "爬坡":         {"type": "feature", "category": "terrain"},
        # 路线节点
        "ROUTE_A": {"type": "route", "name": "环岛路海滨线"},
        "ROUTE_B": {"type": "route", "name": "厦大绿化线"},
        "ROUTE_C": {"type": "route", "name": "五缘湾综合线"},
        "ROUTE_D": {"type": "route", "name": "厦门岛北部路线"},
    },
    "edges": [
        # 路线-地点关系
        ("ROUTE_A", "经过", "白城沙滩"),
        ("ROUTE_A", "经过", "曾厝垵"),
        ("ROUTE_A", "经过", "椰风寨"),
        ("ROUTE_A", "经过", "胡里山炮台"),
        ("ROUTE_B", "经过", "南普陀寺"),
        ("ROUTE_B", "经过", "厦门大学"),
        ("ROUTE_B", "经过", "万石山植物园"),
        ("ROUTE_C", "经过", "五缘湾湿地公园"),
        ("ROUTE_C", "经过", "莲前路"),
        ("ROUTE_D", "经过", "五缘湾北岸"),
        ("ROUTE_D", "经过", "集美大桥头"),
        # 路线-特征关系
        ("ROUTE_A", "具有", "海景"),
        ("ROUTE_A", "具有", "平坦"),
        ("ROUTE_B", "具有", "树荫"),
        ("ROUTE_B", "具有", "软路面"),
        ("ROUTE_B", "具有", "公园"),
        ("ROUTE_C", "具有", "公园"),
        ("ROUTE_C", "具有", "水站"),
        ("ROUTE_D", "具有", "软路面"),
        ("ROUTE_D", "具有", "平坦"),
        # 地点-特征关系
        ("白城沙滩", "具有", "海景"),
        ("曾厝垵", "具有", "海景"),
        ("万石山植物园", "具有", "树荫"),
        ("万石山植物园", "具有", "软路面"),
        ("厦门大学", "具有", "树荫"),
        ("五缘湾湿地公园", "具有", "公园"),
        ("五缘湾北岸", "具有", "软路面"),
        # 地点邻近关系
        ("白城沙滩", "邻近", "厦门大学"),
        ("椰风寨", "邻近", "白城沙滩"),
        ("南普陀寺", "邻近", "万石山植物园"),
        ("五缘湾湿地公园", "邻近", "五缘湾北岸"),
    ]
}


# ============================================================
# 长期记忆系统（基于文件持久化）
# ============================================================
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "user_memory.json")


def load_memory() -> dict:
    """加载用户历史记忆"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "query_history": [],      # 历史查询记录
        "preference_stats": {},   # 偏好统计
        "route_feedback": {},     # 路线反馈
        "session_count": 0,       # 会话次数
    }


def save_memory(memory: dict):
    """保存用户记忆到文件"""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Memory] 保存记忆失败: {e}")


def update_memory(user_query: str, params: dict, recommended_route: str):
    """更新用户记忆"""
    memory = load_memory()
    memory["session_count"] += 1

    # 记录查询历史（最多保留50条）
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "activity_type": params.get("activity_type", "跑步"),
        "duration_min": params.get("duration_min", 60),
        "intensity": params.get("intensity", "中等"),
        "preferred_features": params.get("preferred_features", []),
        "health_constraints": params.get("health_constraints", []),
        "recommended_route": recommended_route,
    }
    memory["query_history"].insert(0, entry)
    memory["query_history"] = memory["query_history"][:50]

    # 更新偏好统计
    for feature in params.get("preferred_features", []):
        memory["preference_stats"][feature] = memory["preference_stats"].get(feature, 0) + 1
    for constraint in params.get("health_constraints", []):
        key = f"constraint_{constraint}"
        memory["preference_stats"][key] = memory["preference_stats"].get(key, 0) + 1

    save_memory(memory)
    return memory


def get_memory_context(params: dict) -> str:
    """根据历史记忆生成上下文提示"""
    memory = load_memory()
    if memory["session_count"] == 0:
        return ""

    context_parts = []

    # 分析偏好趋势
    stats = memory["preference_stats"]
    if stats:
        top_prefs = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        pref_names = {
            "sea_view": "海景", "shade": "树荫", "water": "水站",
            "park": "公园", "scenic": "风景",
            "constraint_ankle": "脚踝保护", "constraint_knee": "膝盖保护"
        }
        top_str = "、".join([pref_names.get(k, k) for k, _ in top_prefs])
        context_parts.append(f"用户历史偏好：{top_str}")

    # 最近3次查询
    recent = memory["query_history"][:3]
    if recent:
        activities = [r["activity_type"] for r in recent]
        most_common = max(set(activities), key=activities.count)
        context_parts.append(f"最近常做运动：{most_common}")

    return "；".join(context_parts) if context_parts else ""


# ============================================================
# TF-IDF 简易实现（不依赖外部库）
# ============================================================
def tokenize(text: str) -> List[str]:
    """简单中文分词（基于关键词列表）"""
    keywords = [
        "海景", "树荫", "软路面", "水站", "公园", "平坦", "爬坡",
        "脚踝", "膝盖", "跑步", "骑行", "徒步", "散步", "快走",
        "白城", "曾厝垵", "椰风寨", "胡里山", "南普陀", "厦大",
        "万石山", "植物园", "五缘湾", "环岛路", "北部", "南部",
        "轻松", "中等", "耐力", "高强度", "短距", "中距", "长距",
        "夏季", "清晨", "傍晚", "补给", "饮水", "安全", "安静",
    ]
    tokens = []
    for kw in keywords:
        if kw in text:
            tokens.append(kw)
    # 也加入单字
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            tokens.append(char)
    return list(set(tokens))


def compute_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """计算查询与文档的余弦相似度"""
    if not query_tokens or not doc_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    intersection = query_set & doc_set
    if not intersection:
        return 0.0
    return len(intersection) / math.sqrt(len(query_set) * len(doc_set))


# ============================================================
# RAG 检索引擎
# ============================================================
def retrieve_knowledge(query: str, params: dict, top_k: int = 3) -> List[Dict]:
    """
    从知识库中检索与查询最相关的知识条目

    Args:
        query: 用户原始查询文本
        params: 解析后的结构化参数
        top_k: 返回最相关的条目数量

    Returns:
        相关知识条目列表，按相关度降序排列
    """
    # 构建查询文本（结合原始查询和结构化参数）
    query_text = query
    if params:
        query_text += " " + " ".join(params.get("preferred_features", []))
        query_text += " " + " ".join(params.get("health_constraints", []))
        query_text += " " + params.get("activity_type", "")
        query_text += " " + params.get("intensity", "")

    query_tokens = tokenize(query_text)

    # 计算每个知识条目的相关度
    scored_docs = []
    for doc in XIAMEN_ROUTE_KNOWLEDGE:
        doc_text = doc["title"] + " " + doc["content"] + " " + " ".join(doc["tags"])
        doc_tokens = tokenize(doc_text)
        score = compute_similarity(query_tokens, doc_tokens)

        # 额外加权：如果文档的路线ID与参数匹配
        if params and doc.get("route_id"):
            preferred = params.get("preferred_features", [])
            constraints = params.get("health_constraints", [])
            if "sea_view" in preferred and doc.get("route_id") == "ROUTE_A":
                score += 0.2
            if "shade" in preferred and doc.get("route_id") == "ROUTE_B":
                score += 0.2
            if ("ankle" in constraints or "knee" in constraints) and doc.get("soft_surface_pct", 0) > 40:
                score += 0.3

        scored_docs.append((score, doc))

    # 按相关度排序，返回top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k] if score > 0]


def get_knowledge_graph_context(location_names: List[str]) -> str:
    """
    基于知识图谱，获取地点相关的上下文信息

    Args:
        location_names: 地点名称列表

    Returns:
        知识图谱上下文字符串
    """
    context_parts = []
    nodes = KNOWLEDGE_GRAPH["nodes"]
    edges = KNOWLEDGE_GRAPH["edges"]

    for location in location_names:
        if location in nodes:
            node = nodes[location]
            # 查找该地点的所有关系
            related = []
            for src, rel, dst in edges:
                if src == location:
                    related.append(f"{rel}{dst}")
                elif dst == location:
                    related.append(f"被{src}{rel}")

            if related:
                context_parts.append(f"{location}：{', '.join(related[:3])}")

    return "；".join(context_parts) if context_parts else ""


def build_rag_context(query: str, params: dict) -> str:
    """
    构建完整的RAG上下文，用于增强LLM的路线描述生成

    Args:
        query: 用户查询
        params: 解析后的参数

    Returns:
        RAG上下文字符串
    """
    context_parts = []

    # 1. 检索相关知识
    relevant_docs = retrieve_knowledge(query, params, top_k=2)
    if relevant_docs:
        for doc in relevant_docs:
            context_parts.append(f"[知识库] {doc['title']}: {doc['content'][:150]}...")

    # 2. 知识图谱上下文
    preferred = params.get("preferred_features", []) if params else []
    locations = []
    if "sea_view" in preferred:
        locations.extend(["白城沙滩", "曾厝垵"])
    if "shade" in preferred:
        locations.extend(["万石山植物园", "厦门大学"])
    if "park" in preferred:
        locations.extend(["五缘湾湿地公园"])

    kg_context = get_knowledge_graph_context(locations)
    if kg_context:
        context_parts.append(f"[知识图谱] {kg_context}")

    # 3. 长期记忆上下文
    memory_context = get_memory_context(params or {})
    if memory_context:
        context_parts.append(f"[用户记忆] {memory_context}")

    return "\n".join(context_parts) if context_parts else ""


# ============================================================
# Agent 智能体规划器
# ============================================================
class RouteAgent:
    """
    运动路线规划Agent
    实现多步推理：需求分析 → 知识检索 → 路线评估 → 个性化推荐
    """

    def __init__(self):
        self.steps_log = []

    def log_step(self, step: str, result: str):
        """记录Agent推理步骤"""
        self.steps_log.append({
            "step": step,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        print(f"[Agent] {step}: {result[:100]}")

    def analyze_user_needs(self, query: str, params: dict) -> dict:
        """步骤1：深度分析用户需求"""
        analysis = {
            "primary_goal": "运动健身",
            "constraints": [],
            "preferences": [],
            "risk_factors": [],
        }

        # 分析健康约束
        constraints = params.get("health_constraints", [])
        if "ankle" in constraints:
            analysis["constraints"].append("脚踝不适，需要软路面")
            analysis["risk_factors"].append("避免硬质路面和坡度")
        if "knee" in constraints:
            analysis["constraints"].append("膝盖不适，需要低冲击路面")
            analysis["risk_factors"].append("避免下坡路段")

        # 分析偏好
        preferred = params.get("preferred_features", [])
        if "sea_view" in preferred:
            analysis["preferences"].append("海景观赏")
        if "shade" in preferred:
            analysis["preferences"].append("树荫遮蔽")
        if "water" in preferred:
            analysis["preferences"].append("补给水站")

        self.log_step("需求分析", f"约束:{analysis['constraints']}, 偏好:{analysis['preferences']}")
        return analysis

    def retrieve_relevant_knowledge(self, query: str, params: dict) -> List[dict]:
        """步骤2：检索相关知识"""
        docs = retrieve_knowledge(query, params, top_k=3)
        self.log_step("知识检索", f"检索到{len(docs)}条相关知识")
        return docs

    def evaluate_routes(self, routes: List[dict], analysis: dict, knowledge: List[dict]) -> List[dict]:
        """步骤3：基于路线属性和用户需求全面评估每条路线，确保每条路线都有完整分析"""
        constraints = analysis.get("constraints", [])
        preferences = analysis.get("preferences", [])
        risk_factors = analysis.get("risk_factors", [])
        has_joint_constraint = bool(constraints)  # 有关节约束
        wants_shade = "树荫遮蔽" in preferences
        wants_sea = "海景观赏" in preferences
        wants_water = "补给水站" in preferences

        for route in routes:
            route_id = route.get("route_id", "")
            bonus = 0.0
            reasoning = []

            # ---- 从路线属性直接提取数据 ----
            soft_pct = route.get("soft_surface_pct", 0)
            shade_pct = route.get("shade_coverage_pct", 0)
            water_cnt = route.get("water_stations", 0)
            if isinstance(water_cnt, list):
                water_cnt = len(water_cnt)
            elev = route.get("elevation_gain_m", 0)
            has_sea = bool(route.get("sea_view_point") or route.get("coastal_ratio", 0) > 20)
            dist_km = route.get("distance_km", 0)
            area = route.get("area", "")

            # ---- 软路面评估（关节约束最重要） ----
            if soft_pct >= 50:
                reasoning.append(f"软路面比例{soft_pct}%，对关节冲击极小，非常适合脚踝不适者")
                bonus += 15
            elif soft_pct >= 30:
                reasoning.append(f"软路面比例{soft_pct}%，混合路面，关节压力适中")
                bonus += 8
            elif soft_pct >= 15:
                reasoning.append(f"软路面比例{soft_pct}%，以硬质路面为主，建议穿缓震跑鞋")
                if has_joint_constraint:
                    bonus -= 5
            else:
                reasoning.append(f"路面以沥青/水泥为主（软路面{soft_pct}%），关节约束者需注意")
                if has_joint_constraint:
                    bonus -= 10

            # ---- 树荫评估 ----
            if shade_pct >= 60:
                reasoning.append(f"树荫覆盖{shade_pct}%，遮阴效果优秀，夏季运动首选")
                if wants_shade:
                    bonus += 12
                else:
                    bonus += 5
            elif shade_pct >= 35:
                reasoning.append(f"树荫覆盖{shade_pct}%，遮阴条件良好")
                if wants_shade:
                    bonus += 6
            elif shade_pct >= 20:
                reasoning.append(f"树荫覆盖{shade_pct}%，遮阴有限，建议清晨或傍晚运动")
            else:
                reasoning.append(f"树荫较少（{shade_pct}%），暴露路段较多，注意防晒")
                if wants_shade:
                    bonus -= 5

            # ---- 水站评估 ----
            if water_cnt >= 5:
                reasoning.append(f"沿途{water_cnt}个补给水站，补给充足，适合长距离训练")
                if wants_water:
                    bonus += 10
                else:
                    bonus += 4
            elif water_cnt >= 3:
                reasoning.append(f"沿途{water_cnt}个水站，补给条件良好")
                if wants_water:
                    bonus += 6
            elif water_cnt >= 1:
                reasoning.append(f"沿途{water_cnt}个水站，建议自带部分饮水")
            else:
                reasoning.append("沿途无固定水站，务必自带充足饮水")
                if wants_water:
                    bonus -= 5

            # ---- 海景评估 ----
            if has_sea:
                reasoning.append("路线经过海滨区域，可欣赏海景，视野开阔")
                if wants_sea:
                    bonus += 12
                else:
                    bonus += 3
            else:
                if wants_sea:
                    reasoning.append("该路线以内陆/公园路段为主，海景观赏机会有限")
                    bonus -= 3

            # ---- 坡度评估 ----
            if elev > 200:
                reasoning.append(f"累计爬升{elev}m，坡度较大，关节约束者需谨慎")
                if has_joint_constraint:
                    bonus -= 12
            elif elev > 80:
                reasoning.append(f"累计爬升{elev}m，有一定起伏，增加训练强度")
                if has_joint_constraint:
                    bonus -= 5
            else:
                reasoning.append(f"地势平坦（爬升仅{elev}m），适合关节不适者稳定配速")
                if has_joint_constraint:
                    bonus += 8

            # ---- 区域特色 ----
            if "北部" in area or route_id == "ROUTE_D":
                reasoning.append("位于厦门岛北部，人流量少，环境安静，适合专注训练")
            elif "南部" in area or route_id == "ROUTE_A":
                reasoning.append("位于厦门岛南部海滨，风景优美，氛围活跃")
            elif "中部" in area or route_id == "ROUTE_B":
                reasoning.append("位于厦门岛中部，绿化丰富，校园氛围浓厚")

            # ---- 距离适配评估 ----
            if dist_km > 0:
                if dist_km >= 12:
                    reasoning.append(f"全程{dist_km}km，属长距离路线，适合耐力训练")
                elif dist_km >= 7:
                    reasoning.append(f"全程{dist_km}km，中等距离，适合大多数训练目标")
                else:
                    reasoning.append(f"全程{dist_km}km，距离适中，适合轻松跑")

            route["agent_bonus"] = round(bonus, 1)
            route["agent_reasoning"] = reasoning
            route["comprehensive_score"] = round(
                route.get("comprehensive_score", route.get("score", 50)) + bonus, 1
            )

        self.log_step("路线评估", f"完成{len(routes)}条路线的Agent评估，每条路线均有{min(len(r.get('agent_reasoning',[])) for r in routes)}+条分析")
        return routes

    def generate_recommendation(self, routes: List[dict], analysis: dict) -> dict:
        """步骤4：生成最终推荐"""
        if not routes:
            return {}

        # 按综合评分排序
        sorted_routes = sorted(routes, key=lambda r: r.get("comprehensive_score", 0), reverse=True)
        best = sorted_routes[0]

        recommendation = {
            "recommended_route_id": best.get("route_id", ""),
            "reasoning": best.get("agent_reasoning", []),
            "agent_steps": self.steps_log,
        }

        self.log_step("生成推荐", f"推荐路线: {best.get('route_id')}，评分: {best.get('comprehensive_score', 0):.1f}")
        return recommendation

    def run(self, query: str, params: dict, routes: List[dict]) -> dict:
        """执行完整的Agent规划流程"""
        self.steps_log = []
        print("\n[Agent] ===== 开始Agent规划流程 =====")

        # 多步推理
        analysis = self.analyze_user_needs(query, params)
        knowledge = self.retrieve_relevant_knowledge(query, params)
        evaluated_routes = self.evaluate_routes(routes, analysis, knowledge)
        recommendation = self.generate_recommendation(evaluated_routes, analysis)

        # 更新长期记忆
        update_memory(query, params, recommendation.get("recommended_route_id", ""))

        print("[Agent] ===== Agent规划完成 =====\n")
        return {
            "routes": evaluated_routes,
            "recommendation": recommendation,
            "rag_context": build_rag_context(query, params),
        }


# 全局Agent实例
route_agent = RouteAgent()
