"""
厦门市GIS数据初始化脚本
将厦门市路网、DEM高程、POI兴趣点、NDVI植被指数数据导入PostgreSQL/PostGIS数据库
研究区域：厦门岛南部（环岛路沿线，含白城、椰风寨、胡里山、曾厝垵等区域）
坐标范围：纬度 24.42~24.46，经度 118.07~118.17
"""
import psycopg2
import json
import math
import random
import os
import sys

# 数据库连接配置
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "dbname": os.environ.get("DB_NAME", "sports_db"),
    "user": os.environ.get("DB_USER", "sports_user"),
    "password": os.environ.get("DB_PASSWORD", "sports_pass123"),
}

# ============================================================
# 厦门岛南部路网节点数据（基于真实地理坐标）
# 覆盖环岛路、白城、椰风寨、胡里山、曾厝垵等区域
# ============================================================
XIAMEN_NODES = [
    # 椰风寨区域
    {"id": 1001, "lat": 24.4380, "lon": 118.0850, "name": "椰风寨公园入口"},
    {"id": 1002, "lat": 24.4360, "lon": 118.0870, "name": "椰风寨海滨步道"},
    {"id": 1003, "lat": 24.4340, "lon": 118.0890, "name": "椰风寨南端"},
    {"id": 1004, "lat": 24.4320, "lon": 118.0910, "name": "灯塔路口"},
    {"id": 1005, "lat": 24.4300, "lon": 118.0930, "name": "灯塔观景台"},
    # 白城沙滩区域
    {"id": 1010, "lat": 24.4420, "lon": 118.0870, "name": "白城沙滩入口"},
    {"id": 1011, "lat": 24.4440, "lon": 118.0890, "name": "白城海滨浴场"},
    {"id": 1012, "lat": 24.4460, "lon": 118.0910, "name": "白城路口"},
    {"id": 1013, "lat": 24.4480, "lon": 118.0930, "name": "白城公交站"},
    # 环岛路主干道（西段）
    {"id": 1020, "lat": 24.4400, "lon": 118.0820, "name": "环岛路西端"},
    {"id": 1021, "lat": 24.4380, "lon": 118.0840, "name": "环岛路西段1"},
    {"id": 1022, "lat": 24.4360, "lon": 118.0860, "name": "环岛路西段2"},
    {"id": 1023, "lat": 24.4340, "lon": 118.0880, "name": "环岛路西段3"},
    {"id": 1024, "lat": 24.4320, "lon": 118.0900, "name": "环岛路西段4"},
    # 环岛路主干道（中段）
    {"id": 1030, "lat": 24.4350, "lon": 118.0950, "name": "环岛路中段1"},
    {"id": 1031, "lat": 24.4370, "lon": 118.0970, "name": "环岛路中段2"},
    {"id": 1032, "lat": 24.4390, "lon": 118.0990, "name": "环岛路中段3"},
    {"id": 1033, "lat": 24.4410, "lon": 118.1010, "name": "环岛路中段4"},
    {"id": 1034, "lat": 24.4430, "lon": 118.1030, "name": "环岛路中段5"},
    # 胡里山炮台区域
    {"id": 1040, "lat": 24.4430, "lon": 118.1050, "name": "胡里山炮台入口"},
    {"id": 1041, "lat": 24.4450, "lon": 118.1070, "name": "胡里山公园"},
    {"id": 1042, "lat": 24.4460, "lon": 118.1090, "name": "胡里山海岸线"},
    {"id": 1043, "lat": 24.4440, "lon": 118.1110, "name": "胡里山东侧步道"},
    # 环岛路主干道（东段）
    {"id": 1050, "lat": 24.4420, "lon": 118.1130, "name": "环岛路东段1"},
    {"id": 1051, "lat": 24.4440, "lon": 118.1150, "name": "环岛路东段2"},
    {"id": 1052, "lat": 24.4460, "lon": 118.1170, "name": "环岛路东段3"},
    {"id": 1053, "lat": 24.4480, "lon": 118.1190, "name": "环岛路东段4"},
    # 曾厝垵区域
    {"id": 1060, "lat": 24.4490, "lon": 118.1210, "name": "曾厝垵西入口"},
    {"id": 1061, "lat": 24.4500, "lon": 118.1230, "name": "曾厝垵文创村"},
    {"id": 1062, "lat": 24.4510, "lon": 118.1250, "name": "曾厝垵中心"},
    {"id": 1063, "lat": 24.4520, "lon": 118.1270, "name": "曾厝垵东出口"},
    # 内陆连接节点
    {"id": 1070, "lat": 24.4500, "lon": 118.0850, "name": "莲前西路口"},
    {"id": 1071, "lat": 24.4500, "lon": 118.0950, "name": "莲前中路口"},
    {"id": 1072, "lat": 24.4500, "lon": 118.1050, "name": "莲前东路口"},
    {"id": 1073, "lat": 24.4500, "lon": 118.1150, "name": "莲前路东段"},
    # 海滨步道（沿海岸线）
    {"id": 1080, "lat": 24.4330, "lon": 118.0920, "name": "海滨步道1"},
    {"id": 1081, "lat": 24.4340, "lon": 118.0960, "name": "海滨步道2"},
    {"id": 1082, "lat": 24.4350, "lon": 118.1000, "name": "海滨步道3"},
    {"id": 1083, "lat": 24.4360, "lon": 118.1040, "name": "海滨步道4"},
    {"id": 1084, "lat": 24.4380, "lon": 118.1080, "name": "海滨步道5"},
    {"id": 1085, "lat": 24.4400, "lon": 118.1120, "name": "海滨步道6"},
    # 公园内部步道
    {"id": 1090, "lat": 24.4410, "lon": 118.0860, "name": "椰风寨公园内道1"},
    {"id": 1091, "lat": 24.4400, "lon": 118.0870, "name": "椰风寨公园内道2"},
    {"id": 1092, "lat": 24.4390, "lon": 118.0880, "name": "椰风寨公园内道3"},
]

# ============================================================
# 厦门路网边数据（连接节点，定义道路属性）
# ============================================================
XIAMEN_EDGES = [
    # 环岛路主干道（双向）
    {"from": 1020, "to": 1021, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1021, "to": 1022, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1022, "to": 1023, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1023, "to": 1024, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1024, "to": 1030, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1030, "to": 1031, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1031, "to": 1032, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1032, "to": 1033, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1033, "to": 1034, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1034, "to": 1040, "length": 250, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1040, "to": 1050, "length": 900, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1050, "to": 1051, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1051, "to": 1052, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1052, "to": 1053, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1053, "to": 1060, "length": 250, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1060, "to": 1061, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1061, "to": 1062, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 1062, "to": 1063, "length": 280, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    # 海滨步道（沿海，软路面，树荫多）
    {"from": 1003, "to": 1080, "length": 350, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1080, "to": 1081, "length": 450, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1081, "to": 1082, "length": 450, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1082, "to": 1083, "length": 450, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1083, "to": 1084, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1084, "to": 1085, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 1085, "to": 1043, "length": 400, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    # 椰风寨公园步道
    {"from": 1001, "to": 1090, "length": 200, "highway": "path", "surface": "ground", "name": "椰风寨步道"},
    {"from": 1090, "to": 1091, "length": 150, "highway": "path", "surface": "ground", "name": "椰风寨步道"},
    {"from": 1091, "to": 1092, "length": 150, "highway": "path", "surface": "ground", "name": "椰风寨步道"},
    {"from": 1092, "to": 1002, "length": 200, "highway": "path", "surface": "ground", "name": "椰风寨步道"},
    {"from": 1002, "to": 1003, "length": 280, "highway": "footway", "surface": "unpaved", "name": "椰风寨海滨"},
    {"from": 1003, "to": 1004, "length": 280, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
    {"from": 1004, "to": 1005, "length": 250, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
    # 白城沙滩步道
    {"from": 1010, "to": 1011, "length": 300, "highway": "footway", "surface": "ground", "name": "白城步道"},
    {"from": 1011, "to": 1012, "length": 280, "highway": "footway", "surface": "ground", "name": "白城步道"},
    {"from": 1012, "to": 1013, "length": 280, "highway": "residential", "surface": "asphalt", "name": "白城路"},
    # 胡里山区域步道
    {"from": 1040, "to": 1041, "length": 300, "highway": "path", "surface": "ground", "name": "胡里山步道"},
    {"from": 1041, "to": 1042, "length": 280, "highway": "footway", "surface": "unpaved", "name": "胡里山海岸"},
    {"from": 1042, "to": 1043, "length": 280, "highway": "footway", "surface": "unpaved", "name": "胡里山海岸"},
    # 莲前路（内陆干道）
    {"from": 1070, "to": 1071, "length": 1100, "highway": "secondary", "surface": "asphalt", "name": "莲前西路"},
    {"from": 1071, "to": 1072, "length": 1100, "highway": "secondary", "surface": "asphalt", "name": "莲前中路"},
    {"from": 1072, "to": 1073, "length": 1100, "highway": "secondary", "surface": "asphalt", "name": "莲前东路"},
    # 连接道路（纵向）
    {"from": 1070, "to": 1010, "length": 900, "highway": "residential", "surface": "asphalt", "name": "白城南路"},
    {"from": 1071, "to": 1033, "length": 700, "highway": "residential", "surface": "asphalt", "name": "环岛路连接"},
    {"from": 1072, "to": 1041, "length": 600, "highway": "residential", "surface": "asphalt", "name": "胡里山路"},
    {"from": 1073, "to": 1053, "length": 600, "highway": "residential", "surface": "asphalt", "name": "曾厝垵路"},
    {"from": 1001, "to": 1020, "length": 350, "highway": "residential", "surface": "asphalt", "name": "椰风寨路"},
    {"from": 1001, "to": 1070, "length": 1300, "highway": "residential", "surface": "asphalt", "name": "南北连接"},
    {"from": 1010, "to": 1021, "length": 600, "highway": "residential", "surface": "asphalt", "name": "白城西路"},
    {"from": 1013, "to": 1071, "length": 700, "highway": "residential", "surface": "asphalt", "name": "白城北路"},
    {"from": 1063, "to": 1073, "length": 800, "highway": "residential", "surface": "asphalt", "name": "曾厝垵北路"},
]

# ============================================================
# 厦门市POI数据（水站、海景点、公园入口等）
# ============================================================
XIAMEN_POI = [
    # 水站
    {"id": "W001", "name": "椰风寨饮水机", "lat": 24.4380, "lon": 118.0855, "type": "water_station", "category": "公园饮水机", "description": "椰风寨公园内免费饮水机"},
    {"id": "W002", "name": "白城沙滩便利店", "lat": 24.4450, "lon": 118.0890, "type": "water_station", "category": "便利店", "description": "白城沙滩旁便利店，可购水"},
    {"id": "W003", "name": "胡里山炮台服务站", "lat": 24.4440, "lon": 118.1055, "type": "water_station", "category": "景区服务站", "description": "胡里山炮台景区内服务站"},
    {"id": "W004", "name": "曾厝垵便利店", "lat": 24.4505, "lon": 118.1235, "type": "water_station", "category": "便利店", "description": "曾厝垵文创村内便利店"},
    {"id": "W005", "name": "环岛路公园饮水机", "lat": 24.4390, "lon": 118.0990, "type": "water_station", "category": "公园饮水机", "description": "环岛路中段公园饮水机"},
    {"id": "W006", "name": "海滨步道补给点", "lat": 24.4360, "lon": 118.1000, "type": "water_station", "category": "补给站", "description": "海滨步道中段补给点"},
    # 海景观景点
    {"id": "S001", "name": "灯塔观景台", "lat": 24.4300, "lon": 118.0930, "type": "sea_view", "category": "观景台", "description": "厦门岛南端灯塔，360度海景"},
    {"id": "S002", "name": "白城海滨观景平台", "lat": 24.4440, "lon": 118.0895, "type": "sea_view", "category": "观景台", "description": "白城沙滩观景平台，视野开阔"},
    {"id": "S003", "name": "胡里山海岸线", "lat": 24.4455, "lon": 118.1090, "type": "sea_view", "category": "海岸线", "description": "胡里山炮台旁海岸线，历史与海景并存"},
    {"id": "S004", "name": "曾厝垵观海台", "lat": 24.4510, "lon": 118.1250, "type": "sea_view", "category": "观景台", "description": "曾厝垵文创村旁观海台"},
    # 公园/绿地
    {"id": "P001", "name": "椰风寨公园", "lat": 24.4385, "lon": 118.0860, "type": "park", "category": "城市公园", "description": "椰风寨滨海公园，绿化好，适合晨跑"},
    {"id": "P002", "name": "胡里山公园", "lat": 24.4450, "lon": 118.1070, "type": "park", "category": "历史公园", "description": "胡里山炮台公园，历史遗迹与绿化结合"},
    {"id": "P003", "name": "曾厝垵文创村", "lat": 24.4505, "lon": 118.1240, "type": "park", "category": "文创园区", "description": "曾厝垵文创村，补给便利"},
    # 厕所/更衣室
    {"id": "T001", "name": "椰风寨公共厕所", "lat": 24.4382, "lon": 118.0858, "type": "toilet", "category": "公共设施", "description": "椰风寨公园内公共厕所"},
    {"id": "T002", "name": "白城沙滩更衣室", "lat": 24.4445, "lon": 118.0888, "type": "toilet", "category": "公共设施", "description": "白城沙滩更衣室及厕所"},
    {"id": "T003", "name": "胡里山景区厕所", "lat": 24.4438, "lon": 118.1052, "type": "toilet", "category": "公共设施", "description": "胡里山景区内厕所"},
]

# ============================================================
# 厦门市DEM高程数据（基于真实地形，环岛路沿线地势平缓）
# ============================================================
def generate_xiamen_dem_data():
    """生成厦门岛南部DEM高程数据（基于真实地形特征）"""
    dem_points = []
    # 厦门岛南部地形：海岸线海拔0-5m，向内陆逐渐升高至20-50m
    lat_start, lat_end = 24.42, 24.46
    lon_start, lon_end = 118.07, 118.17
    step = 0.005  # 约500m分辨率

    lat = lat_start
    while lat <= lat_end:
        lon = lon_start
        while lon <= lon_end:
            # 基于地理位置估算高程
            # 海岸线（纬度低）高程低，内陆高程高
            coastal_factor = (lat - lat_start) / (lat_end - lat_start)
            base_elevation = coastal_factor * 35  # 0-35m
            # 添加地形起伏
            noise = random.gauss(0, 3)
            # 特定区域调整
            if 118.085 <= lon <= 118.095 and 24.43 <= lat <= 24.44:
                base_elevation += 5  # 椰风寨丘陵
            if 118.105 <= lon <= 118.115 and 24.44 <= lat <= 24.45:
                base_elevation += 8  # 胡里山高地
            elevation = max(0, base_elevation + noise)
            dem_points.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "elevation_m": round(elevation, 1)
            })
            lon = round(lon + step, 4)
        lat = round(lat + step, 4)
    return dem_points

# ============================================================
# 厦门市NDVI植被指数数据（基于路段类型和位置）
# ============================================================
NDVI_BY_HIGHWAY_TYPE = {
    "footway": {"base": 0.62, "std": 0.08},   # 步道：植被覆盖好
    "path": {"base": 0.68, "std": 0.07},       # 小径：植被最好
    "track": {"base": 0.65, "std": 0.08},      # 土路：植被好
    "pedestrian": {"base": 0.45, "std": 0.10}, # 行人道：中等
    "residential": {"base": 0.35, "std": 0.10},# 居民路：较少
    "primary": {"base": 0.22, "std": 0.08},    # 主干道：较少
    "secondary": {"base": 0.28, "std": 0.08},  # 次干道：较少
    "tertiary": {"base": 0.32, "std": 0.09},   # 支路：中等
    "cycleway": {"base": 0.55, "std": 0.10},   # 自行车道：较好
}


def create_tables(conn):
    """创建数据库表"""
    cursor = conn.cursor()

    # 路网节点表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS road_nodes (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            geom GEOMETRY(Point, 4326),
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION
        );
    """)

    # 路网边表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS road_edges (
            id SERIAL PRIMARY KEY,
            from_node INTEGER REFERENCES road_nodes(id),
            to_node INTEGER REFERENCES road_nodes(id),
            length_m DOUBLE PRECISION,
            highway VARCHAR(50),
            surface VARCHAR(50),
            road_name VARCHAR(100),
            ndvi DOUBLE PRECISION,
            shade_score DOUBLE PRECISION,
            geom GEOMETRY(LineString, 4326)
        );
    """)

    # POI兴趣点表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS poi_points (
            id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100),
            poi_type VARCHAR(50),
            category VARCHAR(50),
            description TEXT,
            geom GEOMETRY(Point, 4326),
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION
        );
    """)

    # DEM高程表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dem_elevation (
            id SERIAL PRIMARY KEY,
            geom GEOMETRY(Point, 4326),
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,
            elevation_m DOUBLE PRECISION
        );
    """)

    # 创建空间索引
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_road_nodes_geom ON road_nodes USING GIST(geom);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_road_edges_geom ON road_edges USING GIST(geom);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_poi_geom ON poi_points USING GIST(geom);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dem_geom ON dem_elevation USING GIST(geom);")

    conn.commit()
    print("[DB] 数据表创建完成")


def insert_nodes(conn):
    """插入路网节点数据"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM road_nodes;")

    for node in XIAMEN_NODES:
        cursor.execute("""
            INSERT INTO road_nodes (id, name, geom, lat, lon)
            VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                geom = EXCLUDED.geom,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon;
        """, (node["id"], node["name"], node["lon"], node["lat"], node["lat"], node["lon"]))

    conn.commit()
    print(f"[DB] 插入 {len(XIAMEN_NODES)} 个路网节点")


def insert_edges(conn):
    """插入路网边数据（含NDVI计算）"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM road_edges;")

    # 构建节点坐标字典
    node_coords = {n["id"]: (n["lat"], n["lon"]) for n in XIAMEN_NODES}

    for edge in XIAMEN_EDGES:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id not in node_coords or to_id not in node_coords:
            continue

        from_lat, from_lon = node_coords[from_id]
        to_lat, to_lon = node_coords[to_id]

        # 计算NDVI
        highway = edge.get("highway", "residential")
        ndvi_params = NDVI_BY_HIGHWAY_TYPE.get(highway, {"base": 0.30, "std": 0.08})
        ndvi = min(0.90, max(0.05, random.gauss(ndvi_params["base"], ndvi_params["std"])))
        shade_score = ndvi * 100

        # 插入双向边
        for f, t, fl, tl in [(from_id, to_id, from_lat, from_lon), (to_id, from_id, to_lat, to_lon)]:
            cursor.execute("""
                INSERT INTO road_edges (from_node, to_node, length_m, highway, surface, road_name, ndvi, shade_score, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakeLine(
                        ST_MakePoint(%s, %s),
                        ST_MakePoint(%s, %s)
                    ), 4326))
            """, (f, t, edge["length"], highway, edge.get("surface", "asphalt"),
                  edge.get("name", ""), round(ndvi, 3), round(shade_score, 1),
                  node_coords[f][1], node_coords[f][0],
                  node_coords[t][1], node_coords[t][0]))

    conn.commit()
    print(f"[DB] 插入 {len(XIAMEN_EDGES) * 2} 条路网边（双向）")


def insert_poi(conn):
    """插入POI数据"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM poi_points;")

    for poi in XIAMEN_POI:
        cursor.execute("""
            INSERT INTO poi_points (id, name, poi_type, category, description, geom, lat, lon)
            VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                geom = EXCLUDED.geom;
        """, (poi["id"], poi["name"], poi["type"], poi["category"], poi["description"],
              poi["lon"], poi["lat"], poi["lat"], poi["lon"]))

    conn.commit()
    print(f"[DB] 插入 {len(XIAMEN_POI)} 个POI点")


def insert_dem(conn):
    """插入DEM高程数据"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dem_elevation;")

    dem_data = generate_xiamen_dem_data()
    for point in dem_data:
        cursor.execute("""
            INSERT INTO dem_elevation (geom, lat, lon, elevation_m)
            VALUES (ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
        """, (point["lon"], point["lat"], point["lat"], point["lon"], point["elevation_m"]))

    conn.commit()
    print(f"[DB] 插入 {len(dem_data)} 个DEM高程点")


def verify_data(conn):
    """验证数据插入结果"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM road_nodes;")
    nodes_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM road_edges;")
    edges_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM poi_points;")
    poi_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM dem_elevation;")
    dem_count = cursor.fetchone()[0]

    print(f"\n[DB] 数据验证结果:")
    print(f"  路网节点: {nodes_count} 个")
    print(f"  路网边: {edges_count} 条")
    print(f"  POI点: {poi_count} 个")
    print(f"  DEM高程点: {dem_count} 个")

    # 测试PostGIS空间查询
    cursor.execute("""
        SELECT name, ST_AsText(geom)
        FROM road_nodes
        ORDER BY ST_Distance(geom, ST_SetSRID(ST_MakePoint(118.0850, 24.4380), 4326))
        LIMIT 3;
    """)
    nearest = cursor.fetchall()
    print(f"\n[DB] 距椰风寨最近的3个节点:")
    for row in nearest:
        print(f"  {row[0]}: {row[1]}")


if __name__ == "__main__":
    print("=" * 60)
    print("厦门市GIS数据初始化脚本")
    print("=" * 60)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"[DB] 成功连接到数据库: {DB_CONFIG['dbname']}")

        create_tables(conn)
        insert_nodes(conn)
        insert_edges(conn)
        insert_poi(conn)
        insert_dem(conn)
        verify_data(conn)

        conn.close()
        print("\n[DB] 数据初始化完成！")

    except Exception as e:
        print(f"[DB] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
