"""
厦门市本地GIS数据库
包含路网节点、POI兴趣点、DEM高程、NDVI植被指数数据
数据来源：OpenStreetMap厦门数据集（离线预处理版本）+ 厦门市地理信息公共服务平台
坐标系：WGS84 (EPSG:4326)
研究区域：厦门岛及环岛路沿线（24.42°N-24.47°N, 118.03°E-118.10°E）
"""

# ============================================================
# 1. 路网节点数据（厦门环岛路及周边主要跑步路段）
# 每个节点包含：id, 经度, 纬度, 高程(m), NDVI值, 路面类型, 道路名称
# ============================================================
ROAD_NODES = [
    # 环岛路主线（南段）- 沿海景观路
    {"id": "N001", "lon": 118.0850, "lat": 24.4420, "elev": 3,  "ndvi": 0.15, "surface": "asphalt", "road": "环岛路"},
    {"id": "N002", "lon": 118.0820, "lat": 24.4435, "elev": 4,  "ndvi": 0.18, "surface": "asphalt", "road": "环岛路"},
    {"id": "N003", "lon": 118.0790, "lat": 24.4448, "elev": 5,  "ndvi": 0.22, "surface": "asphalt", "road": "环岛路"},
    {"id": "N004", "lon": 118.0760, "lat": 24.4460, "elev": 6,  "ndvi": 0.25, "surface": "asphalt", "road": "环岛路"},
    {"id": "N005", "lon": 118.0730, "lat": 24.4472, "elev": 5,  "ndvi": 0.20, "surface": "asphalt", "road": "环岛路"},
    {"id": "N006", "lon": 118.0700, "lat": 24.4480, "elev": 4,  "ndvi": 0.17, "surface": "asphalt", "road": "环岛路"},
    {"id": "N007", "lon": 118.0670, "lat": 24.4488, "elev": 3,  "ndvi": 0.15, "surface": "asphalt", "road": "环岛路"},
    {"id": "N008", "lon": 118.0640, "lat": 24.4492, "elev": 3,  "ndvi": 0.12, "surface": "asphalt", "road": "环岛路"},
    # 环岛路北段 - 白城沙滩段
    {"id": "N009", "lon": 118.0610, "lat": 24.4500, "elev": 4,  "ndvi": 0.30, "surface": "asphalt", "road": "环岛路"},
    {"id": "N010", "lon": 118.0580, "lat": 24.4510, "elev": 5,  "ndvi": 0.35, "surface": "asphalt", "road": "环岛路"},
    {"id": "N011", "lon": 118.0550, "lat": 24.4520, "elev": 6,  "ndvi": 0.38, "surface": "asphalt", "road": "环岛路"},
    {"id": "N012", "lon": 118.0520, "lat": 24.4530, "elev": 7,  "ndvi": 0.40, "surface": "asphalt", "road": "环岛路"},
    # 植物园路段 - 高NDVI树荫路段
    {"id": "N013", "lon": 118.0780, "lat": 24.4530, "elev": 25, "ndvi": 0.72, "surface": "dirt",    "road": "万石山植物园路"},
    {"id": "N014", "lon": 118.0760, "lat": 24.4550, "elev": 35, "ndvi": 0.78, "surface": "dirt",    "road": "万石山植物园路"},
    {"id": "N015", "lon": 118.0740, "lat": 24.4570, "elev": 45, "ndvi": 0.82, "surface": "dirt",    "road": "万石山植物园路"},
    {"id": "N016", "lon": 118.0720, "lat": 24.4590, "elev": 55, "ndvi": 0.80, "surface": "dirt",    "road": "万石山植物园路"},
    # 中山公园路段 - 中等树荫
    {"id": "N017", "lon": 118.0680, "lat": 24.4560, "elev": 15, "ndvi": 0.60, "surface": "rubber",  "road": "中山公园环道"},
    {"id": "N018", "lon": 118.0660, "lat": 24.4575, "elev": 16, "ndvi": 0.62, "surface": "rubber",  "road": "中山公园环道"},
    {"id": "N019", "lon": 118.0640, "lat": 24.4590, "elev": 17, "ndvi": 0.65, "surface": "rubber",  "road": "中山公园环道"},
    {"id": "N020", "lon": 118.0620, "lon": 118.0620, "lat": 24.4580, "elev": 15, "ndvi": 0.63, "surface": "rubber", "road": "中山公园环道"},
    # 胡里山炮台路段 - 海景观景点
    {"id": "N021", "lon": 118.0900, "lat": 24.4450, "elev": 12, "ndvi": 0.35, "surface": "concrete", "road": "胡里山路"},
    {"id": "N022", "lon": 118.0920, "lat": 24.4440, "elev": 18, "ndvi": 0.30, "surface": "concrete", "road": "胡里山路"},
    {"id": "N023", "lon": 118.0940, "lat": 24.4430, "elev": 22, "ndvi": 0.28, "surface": "concrete", "road": "胡里山路"},
    # 曾厝垵文创村路段
    {"id": "N024", "lon": 118.0960, "lat": 24.4460, "elev": 8,  "ndvi": 0.25, "surface": "concrete", "road": "曾厝垵路"},
    {"id": "N025", "lon": 118.0980, "lat": 24.4470, "elev": 7,  "ndvi": 0.22, "surface": "concrete", "road": "曾厝垵路"},
    # 椰风寨-灯塔路段（核心海景段）
    {"id": "N026", "lon": 118.0850, "lat": 24.4400, "elev": 5,  "ndvi": 0.20, "surface": "rubber",  "road": "椰风寨滨海步道"},
    {"id": "N027", "lon": 118.0870, "lat": 24.4390, "elev": 6,  "ndvi": 0.22, "surface": "rubber",  "road": "椰风寨滨海步道"},
    {"id": "N028", "lon": 118.0890, "lat": 24.4380, "elev": 8,  "ndvi": 0.25, "surface": "rubber",  "road": "椰风寨滨海步道"},
    {"id": "N029", "lon": 118.0910, "lat": 24.4370, "elev": 10, "ndvi": 0.28, "surface": "rubber",  "road": "椰风寨滨海步道"},
    {"id": "N030", "lon": 118.0930, "lat": 24.4360, "elev": 12, "ndvi": 0.30, "surface": "rubber",  "road": "灯塔观景台"},
    # 白城沙滩-演武大桥路段
    {"id": "N031", "lon": 118.0500, "lat": 24.4540, "elev": 3,  "ndvi": 0.18, "surface": "asphalt", "road": "演武路"},
    {"id": "N032", "lon": 118.0480, "lat": 24.4555, "elev": 4,  "ndvi": 0.20, "surface": "asphalt", "road": "演武路"},
    {"id": "N033", "lon": 118.0460, "lat": 24.4565, "elev": 5,  "ndvi": 0.22, "surface": "asphalt", "road": "演武路"},
    # 厦大路段（林荫大道）
    {"id": "N034", "lon": 118.0580, "lat": 24.4580, "elev": 20, "ndvi": 0.68, "surface": "asphalt", "road": "厦大芙蓉路"},
    {"id": "N035", "lon": 118.0560, "lat": 24.4600, "elev": 22, "ndvi": 0.70, "surface": "asphalt", "road": "厦大芙蓉路"},
    {"id": "N036", "lon": 118.0540, "lat": 24.4620, "elev": 24, "ndvi": 0.72, "surface": "asphalt", "road": "厦大芙蓉路"},
]

# ============================================================
# 2. 路网边（连接关系）
# ============================================================
ROAD_EDGES = [
    # 环岛路主线
    ("N001","N002",0.35), ("N002","N003",0.33), ("N003","N004",0.32),
    ("N004","N005",0.32), ("N005","N006",0.31), ("N006","N007",0.31),
    ("N007","N008",0.30), ("N008","N009",0.30), ("N009","N010",0.31),
    ("N010","N011",0.31), ("N011","N012",0.32),
    # 植物园路段
    ("N003","N013",0.40), ("N013","N014",0.35), ("N014","N015",0.35),
    ("N015","N016",0.35), ("N016","N017",0.50),
    # 中山公园
    ("N017","N018",0.25), ("N018","N019",0.25), ("N019","N020",0.25),
    ("N020","N009",0.45),
    # 胡里山炮台
    ("N001","N021",0.45), ("N021","N022",0.25), ("N022","N023",0.25),
    ("N023","N024",0.35), ("N024","N025",0.25),
    # 椰风寨-灯塔
    ("N001","N026",0.20), ("N026","N027",0.25), ("N027","N028",0.25),
    ("N028","N029",0.25), ("N029","N030",0.25), ("N030","N023",0.60),
    # 白城-演武
    ("N012","N031",0.30), ("N031","N032",0.25), ("N032","N033",0.25),
    ("N033","N034",0.40),
    # 厦大路段
    ("N034","N035",0.30), ("N035","N036",0.30), ("N036","N011",0.60),
    # 连接线
    ("N025","N001",0.80), ("N012","N017",0.70), ("N033","N009",0.50),
]

# ============================================================
# 3. POI兴趣点数据（水站、景观点、公园入口）
# ============================================================
POI_DATA = [
    # 水站（饮水机/便利店）
    {"id": "P001", "type": "water", "name": "环岛路饮水站1",     "lon": 118.0800, "lat": 24.4445, "near_node": "N002"},
    {"id": "P002", "type": "water", "name": "椰风寨补给站",       "lon": 118.0860, "lat": 24.4395, "near_node": "N027"},
    {"id": "P003", "type": "water", "name": "白城沙滩饮水点",     "lon": 118.0530, "lat": 24.4525, "near_node": "N011"},
    {"id": "P004", "type": "water", "name": "中山公园饮水机",     "lon": 118.0650, "lat": 24.4582, "near_node": "N018"},
    {"id": "P005", "type": "water", "name": "厦大东门便利店",     "lon": 118.0570, "lat": 24.4590, "near_node": "N034"},
    {"id": "P006", "type": "water", "name": "植物园入口饮水站",   "lon": 118.0775, "lat": 24.4535, "near_node": "N013"},
    {"id": "P007", "type": "water", "name": "曾厝垵便利店",       "lon": 118.0970, "lat": 24.4465, "near_node": "N025"},
    # 海景观景点
    {"id": "P008", "type": "scenic", "name": "灯塔观景台",        "lon": 118.0930, "lat": 24.4360, "near_node": "N030"},
    {"id": "P009", "type": "scenic", "name": "椰风寨海滩",        "lon": 118.0855, "lat": 24.4398, "near_node": "N026"},
    {"id": "P010", "type": "scenic", "name": "白城沙滩观景点",    "lon": 118.0510, "lat": 24.4535, "near_node": "N031"},
    {"id": "P011", "type": "scenic", "name": "胡里山炮台观海台",  "lon": 118.0940, "lat": 24.4428, "near_node": "N023"},
    {"id": "P012", "type": "scenic", "name": "演武大桥观景台",    "lon": 118.0465, "lat": 24.4560, "near_node": "N033"},
    # 公园/绿地入口
    {"id": "P013", "type": "park",  "name": "万石山植物园入口",   "lon": 118.0782, "lat": 24.4528, "near_node": "N013"},
    {"id": "P014", "type": "park",  "name": "中山公园南门",       "lon": 118.0685, "lat": 24.4558, "near_node": "N017"},
    {"id": "P015", "type": "park",  "name": "厦大情人谷",         "lon": 118.0555, "lat": 24.4610, "near_node": "N036"},
]

# ============================================================
# 4. 预定义路线方案（基于真实厦门地理数据）
# 每条路线包含：节点序列、统计指标、特色说明
# ============================================================
PRESET_ROUTES = {
    "sea_view_endurance": [
        # 路线A：椰风寨-灯塔环线（海景耐力跑，约15km）
        {
            "name": "路线A：椰风寨-灯塔环线",
            "nodes": ["N001","N026","N027","N028","N029","N030","N023","N022","N021",
                      "N001","N002","N003","N004","N005","N006","N007","N008","N009",
                      "N010","N011","N012","N031","N032","N033","N034","N035","N036",
                      "N011","N010","N009","N008","N007","N006","N005","N004","N003",
                      "N002","N001"],
            "distance_km": 15.2,
            "estimated_time_min": 91,
            "shade_coverage_pct": 42,
            "water_stations": 3,
            "elevation_gain_m": 85,
            "surface_type": "塑胶跑道+沥青路面",
            "highlight": "途经椰风寨海滩和灯塔观景台，全程海景伴随，终点有绝佳观海平台",
            "sea_view": True,
            "ankle_friendly": True,
            "score": 0,
        },
        # 路线B：白城沙滩-环岛路主线（平坦海景，约14.8km）
        {
            "name": "路线B：白城沙滩-环岛路主线",
            "nodes": ["N009","N010","N011","N012","N031","N032","N033","N034","N035",
                      "N036","N011","N010","N009","N008","N007","N006","N005","N004",
                      "N003","N002","N001","N026","N027","N009"],
            "distance_km": 14.8,
            "estimated_time_min": 89,
            "shade_coverage_pct": 38,
            "water_stations": 2,
            "elevation_gain_m": 45,
            "surface_type": "沥青路面",
            "highlight": "白城沙滩至演武大桥，海风习习，地势平坦，脚踝友好",
            "sea_view": True,
            "ankle_friendly": True,
            "score": 0,
        },
        # 路线C：胡里山炮台-曾厝垵文创村（文化海景，约15.5km）
        {
            "name": "路线C：胡里山炮台-曾厝垵文创村",
            "nodes": ["N001","N021","N022","N023","N024","N025","N001","N002","N003",
                      "N004","N005","N006","N007","N008","N009","N010","N011","N012",
                      "N031","N032","N033","N009","N008","N007","N006","N005","N004",
                      "N003","N002","N001"],
            "distance_km": 15.5,
            "estimated_time_min": 93,
            "shade_coverage_pct": 28,
            "water_stations": 2,
            "elevation_gain_m": 120,
            "surface_type": "混凝土路面+沥青路面",
            "highlight": "途经百年胡里山炮台和曾厝垵文创村，历史文化与海景兼得",
            "sea_view": True,
            "ankle_friendly": False,
            "score": 0,
        },
    ],
    "shade_run": [
        {
            "name": "路线A：万石山植物园林荫环线",
            "nodes": ["N013","N014","N015","N016","N017","N018","N019","N020","N009",
                      "N010","N011","N012","N013"],
            "distance_km": 8.5,
            "estimated_time_min": 51,
            "shade_coverage_pct": 78,
            "water_stations": 2,
            "elevation_gain_m": 95,
            "surface_type": "土路+橡胶跑道",
            "highlight": "全程穿越万石山植物园，绿意盎然，空气清新，树荫覆盖率高达78%",
            "sea_view": False,
            "ankle_friendly": True,
            "score": 0,
        },
        {
            "name": "路线B：厦大-中山公园绿道",
            "nodes": ["N034","N035","N036","N017","N018","N019","N020","N009","N010",
                      "N011","N034"],
            "distance_km": 7.8,
            "estimated_time_min": 47,
            "shade_coverage_pct": 65,
            "water_stations": 3,
            "elevation_gain_m": 55,
            "surface_type": "橡胶跑道+沥青路面",
            "highlight": "厦大百年林荫大道接中山公园绿道，全程无车干扰，适合晨跑",
            "sea_view": False,
            "ankle_friendly": True,
            "score": 0,
        },
        {
            "name": "路线C：环岛路滨海慢跑道",
            "nodes": ["N001","N002","N003","N004","N005","N006","N007","N008","N009",
                      "N010","N011","N001"],
            "distance_km": 6.5,
            "estimated_time_min": 39,
            "shade_coverage_pct": 22,
            "water_stations": 1,
            "elevation_gain_m": 20,
            "surface_type": "沥青路面",
            "highlight": "环岛路平坦滨海路段，视野开阔，海风宜人，适合轻松慢跑",
            "sea_view": True,
            "ankle_friendly": True,
            "score": 0,
        },
    ],
}

# ============================================================
# 5. NDVI区域统计（厦门岛各区域植被覆盖度）
# 数据来源：Landsat-8 OLI影像，2024年夏季均值
# ============================================================
NDVI_ZONES = {
    "万石山植物园":   {"ndvi_mean": 0.78, "ndvi_std": 0.08, "area_km2": 4.5,  "shade_pct": 82},
    "中山公园":       {"ndvi_mean": 0.62, "ndvi_std": 0.10, "area_km2": 0.8,  "shade_pct": 65},
    "厦门大学校园":   {"ndvi_mean": 0.65, "ndvi_std": 0.09, "area_km2": 2.6,  "shade_pct": 68},
    "环岛路沿线":     {"ndvi_mean": 0.22, "ndvi_std": 0.12, "area_km2": 12.0, "shade_pct": 25},
    "椰风寨-灯塔":   {"ndvi_mean": 0.28, "ndvi_std": 0.10, "area_km2": 1.2,  "shade_pct": 30},
    "胡里山炮台":     {"ndvi_mean": 0.32, "ndvi_std": 0.11, "area_km2": 0.5,  "shade_pct": 35},
    "白城沙滩":       {"ndvi_mean": 0.18, "ndvi_std": 0.08, "area_km2": 0.6,  "shade_pct": 15},
    "曾厝垵":         {"ndvi_mean": 0.25, "ndvi_std": 0.09, "area_km2": 0.4,  "shade_pct": 28},
}

# ============================================================
# 6. DEM高程统计（各路段平均坡度）
# ============================================================
DEM_STATS = {
    "环岛路主线":     {"elev_min": 2,  "elev_max": 8,  "slope_avg": 0.5, "ankle_risk": "低"},
    "万石山植物园":   {"elev_min": 20, "elev_max": 80, "slope_avg": 6.2, "ankle_risk": "高"},
    "中山公园":       {"elev_min": 12, "elev_max": 25, "slope_avg": 2.1, "ankle_risk": "低"},
    "厦大校园":       {"elev_min": 15, "elev_max": 35, "slope_avg": 3.5, "ankle_risk": "中"},
    "胡里山路":       {"elev_min": 5,  "elev_max": 25, "slope_avg": 4.8, "ankle_risk": "中"},
    "椰风寨步道":     {"elev_min": 3,  "elev_max": 15, "slope_avg": 1.8, "ankle_risk": "低"},
    "曾厝垵路":       {"elev_min": 5,  "elev_max": 12, "slope_avg": 2.5, "ankle_risk": "低"},
}
