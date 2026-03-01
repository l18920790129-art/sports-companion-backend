"""
扩充厦门市路网数据
增加更多节点，覆盖完整的厦门岛南部环岛路（约20km环线）
确保路线能达到90分钟耐力跑的合理距离（12-15km）
"""
import psycopg2
import psycopg2.extras
import os
import random
import math

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "dbname": os.environ.get("DB_NAME", "sports_db"),
    "user": os.environ.get("DB_USER", "sports_user"),
    "password": os.environ.get("DB_PASSWORD", "sports_pass123"),
}

# 扩充的节点：覆盖完整环岛路（约20km）
# 厦门岛南部环岛路实际走向：从椰风寨→白城→胡里山→曾厝垵→黄厝→前埔→五缘湾
EXTRA_NODES = [
    # 黄厝区域（曾厝垵以东）
    {"id": 2001, "lat": 24.4530, "lon": 118.1300, "name": "黄厝海滩入口"},
    {"id": 2002, "lat": 24.4540, "lon": 118.1350, "name": "黄厝沙滩"},
    {"id": 2003, "lat": 24.4550, "lon": 118.1400, "name": "黄厝观景台"},
    {"id": 2004, "lat": 24.4545, "lon": 118.1450, "name": "黄厝东段"},
    {"id": 2005, "lat": 24.4540, "lon": 118.1500, "name": "前埔西入口"},
    # 前埔区域
    {"id": 2010, "lat": 24.4535, "lon": 118.1550, "name": "前埔海滨"},
    {"id": 2011, "lat": 24.4530, "lon": 118.1600, "name": "前埔中段"},
    {"id": 2012, "lat": 24.4525, "lon": 118.1650, "name": "前埔东段"},
    {"id": 2013, "lat": 24.4520, "lon": 118.1700, "name": "前埔东端"},
    # 五缘湾区域
    {"id": 2020, "lat": 24.4515, "lon": 118.1750, "name": "五缘湾西入口"},
    {"id": 2021, "lat": 24.4520, "lon": 118.1800, "name": "五缘湾南侧"},
    {"id": 2022, "lat": 24.4530, "lon": 118.1850, "name": "五缘湾湿地公园"},
    # 环岛路北段（连接各区域）
    {"id": 2030, "lat": 24.4560, "lon": 118.1300, "name": "环岛路北段1"},
    {"id": 2031, "lat": 24.4570, "lon": 118.1200, "name": "环岛路北段2"},
    {"id": 2032, "lat": 24.4565, "lon": 118.1100, "name": "环岛路北段3"},
    {"id": 2033, "lat": 24.4560, "lon": 118.1000, "name": "环岛路北段4"},
    {"id": 2034, "lat": 24.4555, "lon": 118.0900, "name": "环岛路北段5"},
    # 海滨步道延伸（连接各海滩）
    {"id": 2040, "lat": 24.4330, "lon": 118.1160, "name": "海滨步道东段1"},
    {"id": 2041, "lat": 24.4340, "lon": 118.1200, "name": "海滨步道东段2"},
    {"id": 2042, "lat": 24.4360, "lon": 118.1240, "name": "海滨步道东段3"},
    {"id": 2043, "lat": 24.4390, "lon": 118.1270, "name": "海滨步道东段4"},
    # 内陆连接（莲前路延伸）
    {"id": 2050, "lat": 24.4500, "lon": 118.1300, "name": "莲前路东延1"},
    {"id": 2051, "lat": 24.4500, "lon": 118.1450, "name": "莲前路东延2"},
    {"id": 2052, "lat": 24.4500, "lon": 118.1600, "name": "莲前路东延3"},
    # 公园步道（增加软路面选项）
    {"id": 2060, "lat": 24.4420, "lon": 118.0840, "name": "海湾公园步道1"},
    {"id": 2061, "lat": 24.4430, "lon": 118.0860, "name": "海湾公园步道2"},
    {"id": 2062, "lat": 24.4440, "lon": 118.0880, "name": "海湾公园步道3"},
    {"id": 2063, "lat": 24.4450, "lon": 118.0900, "name": "海湾公园步道4"},
    # 灯塔区域延伸
    {"id": 2070, "lat": 24.4280, "lon": 118.0950, "name": "灯塔南端"},
    {"id": 2071, "lat": 24.4270, "lon": 118.0970, "name": "灯塔最南端"},
    {"id": 2072, "lat": 24.4290, "lon": 118.0990, "name": "灯塔东侧"},
]

EXTRA_EDGES = [
    # 黄厝区域连接
    {"from": 1063, "to": 2001, "length": 350, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2001, "to": 2002, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2002, "to": 2003, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2003, "to": 2004, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2004, "to": 2005, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    # 黄厝海滨步道
    {"from": 2043, "to": 2001, "length": 350, "highway": "footway", "surface": "ground", "name": "黄厝步道"},
    {"from": 2001, "to": 2002, "length": 600, "highway": "footway", "surface": "ground", "name": "黄厝步道"},
    # 前埔区域
    {"from": 2005, "to": 2010, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2010, "to": 2011, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2011, "to": 2012, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2012, "to": 2013, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    # 五缘湾
    {"from": 2013, "to": 2020, "length": 600, "highway": "primary", "surface": "asphalt", "name": "环岛路"},
    {"from": 2020, "to": 2021, "length": 600, "highway": "footway", "surface": "ground", "name": "五缘湾步道"},
    {"from": 2021, "to": 2022, "length": 600, "highway": "footway", "surface": "ground", "name": "五缘湾步道"},
    # 环岛路北段（形成大环线）
    {"from": 2001, "to": 2030, "length": 400, "highway": "residential", "surface": "asphalt", "name": "黄厝北路"},
    {"from": 2030, "to": 2031, "length": 1200, "highway": "secondary", "surface": "asphalt", "name": "环岛路北段"},
    {"from": 2031, "to": 2032, "length": 1200, "highway": "secondary", "surface": "asphalt", "name": "环岛路北段"},
    {"from": 2032, "to": 2033, "length": 1200, "highway": "secondary", "surface": "asphalt", "name": "环岛路北段"},
    {"from": 2033, "to": 2034, "length": 1200, "highway": "secondary", "surface": "asphalt", "name": "环岛路北段"},
    {"from": 2034, "to": 1070, "length": 800, "highway": "secondary", "surface": "asphalt", "name": "环岛路北段"},
    # 海滨步道东段
    {"from": 1085, "to": 2040, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 2040, "to": 2041, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 2041, "to": 2042, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 2042, "to": 2043, "length": 500, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    {"from": 2043, "to": 1060, "length": 400, "highway": "footway", "surface": "unpaved", "name": "海滨步道"},
    # 莲前路东延
    {"from": 1073, "to": 2050, "length": 1200, "highway": "secondary", "surface": "asphalt", "name": "莲前东路"},
    {"from": 2050, "to": 2051, "length": 1500, "highway": "secondary", "surface": "asphalt", "name": "莲前东路"},
    {"from": 2051, "to": 2052, "length": 1500, "highway": "secondary", "surface": "asphalt", "name": "莲前东路"},
    {"from": 2050, "to": 2001, "length": 700, "highway": "residential", "surface": "asphalt", "name": "黄厝路"},
    {"from": 2051, "to": 2005, "length": 700, "highway": "residential", "surface": "asphalt", "name": "前埔路"},
    {"from": 2052, "to": 2013, "length": 700, "highway": "residential", "surface": "asphalt", "name": "前埔东路"},
    # 公园步道
    {"from": 1070, "to": 2060, "length": 400, "highway": "path", "surface": "ground", "name": "海湾公园"},
    {"from": 2060, "to": 2061, "length": 300, "highway": "path", "surface": "ground", "name": "海湾公园"},
    {"from": 2061, "to": 2062, "length": 300, "highway": "path", "surface": "ground", "name": "海湾公园"},
    {"from": 2062, "to": 2063, "length": 300, "highway": "path", "surface": "ground", "name": "海湾公园"},
    {"from": 2063, "to": 1013, "length": 400, "highway": "path", "surface": "ground", "name": "海湾公园"},
    {"from": 2061, "to": 1021, "length": 350, "highway": "residential", "surface": "asphalt", "name": "连接路"},
    # 灯塔区域延伸
    {"from": 1005, "to": 2070, "length": 300, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
    {"from": 2070, "to": 2071, "length": 300, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
    {"from": 2071, "to": 2072, "length": 350, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
    {"from": 2072, "to": 1080, "length": 400, "highway": "footway", "surface": "unpaved", "name": "灯塔步道"},
]

# 扩充的POI
EXTRA_POI = [
    {"id": "W007", "name": "黄厝海滩补给站", "lat": 24.4535, "lon": 118.1355, "type": "water_station", "category": "补给站", "description": "黄厝海滩旁补给站"},
    {"id": "W008", "name": "前埔便利店", "lat": 24.4530, "lon": 118.1605, "type": "water_station", "category": "便利店", "description": "前埔海滨便利店"},
    {"id": "W009", "name": "五缘湾服务站", "lat": 24.4520, "lon": 118.1805, "type": "water_station", "category": "服务站", "description": "五缘湾湿地公园服务站"},
    {"id": "S005", "name": "黄厝观景台", "lat": 24.4550, "lon": 118.1405, "type": "sea_view", "category": "观景台", "description": "黄厝海滩观景台，视野开阔"},
    {"id": "S006", "name": "五缘湾观海台", "lat": 24.4525, "lon": 118.1855, "type": "sea_view", "category": "观景台", "description": "五缘湾湿地公园观海台"},
    {"id": "P004", "name": "黄厝海滩公园", "lat": 24.4540, "lon": 118.1405, "type": "park", "category": "海滨公园", "description": "黄厝海滩公园，沙滩跑道"},
    {"id": "P005", "name": "五缘湾湿地公园", "lat": 24.4525, "lon": 118.1805, "type": "park", "category": "湿地公园", "description": "五缘湾湿地公园，生态步道"},
]

NDVI_BY_HIGHWAY_TYPE = {
    "footway": {"base": 0.62, "std": 0.08},
    "path": {"base": 0.68, "std": 0.07},
    "track": {"base": 0.65, "std": 0.08},
    "pedestrian": {"base": 0.45, "std": 0.10},
    "residential": {"base": 0.35, "std": 0.10},
    "primary": {"base": 0.22, "std": 0.08},
    "secondary": {"base": 0.28, "std": 0.08},
    "tertiary": {"base": 0.32, "std": 0.09},
}


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # 获取现有节点坐标
    cursor.execute("SELECT id, lat, lon FROM road_nodes;")
    existing_nodes = {row["id"]: (row["lat"], row["lon"]) for row in cursor.fetchall()}

    # 插入新节点
    for node in EXTRA_NODES:
        cursor.execute("""
            INSERT INTO road_nodes (id, name, geom, lat, lon)
            VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (node["id"], node["name"], node["lon"], node["lat"], node["lat"], node["lon"]))
        existing_nodes[node["id"]] = (node["lat"], node["lon"])

    conn.commit()
    print(f"[DB] 插入 {len(EXTRA_NODES)} 个新节点")

    # 插入新边
    edge_count = 0
    for edge in EXTRA_EDGES:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id not in existing_nodes or to_id not in existing_nodes:
            print(f"[WARN] 节点不存在: {from_id} -> {to_id}")
            continue

        highway = edge.get("highway", "residential")
        ndvi_params = NDVI_BY_HIGHWAY_TYPE.get(highway, {"base": 0.30, "std": 0.08})
        ndvi = min(0.90, max(0.05, random.gauss(ndvi_params["base"], ndvi_params["std"])))

        for f, t in [(from_id, to_id), (to_id, from_id)]:
            f_lat, f_lon = existing_nodes[f]
            t_lat, t_lon = existing_nodes[t]
            cursor.execute("""
                INSERT INTO road_edges (from_node, to_node, length_m, highway, surface, road_name, ndvi, shade_score, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakeLine(
                        ST_MakePoint(%s, %s),
                        ST_MakePoint(%s, %s)
                    ), 4326))
            """, (f, t, edge["length"], highway, edge.get("surface", "asphalt"),
                  edge.get("name", ""), round(ndvi, 3), round(ndvi * 100, 1),
                  f_lon, f_lat, t_lon, t_lat))
            edge_count += 1

    conn.commit()
    print(f"[DB] 插入 {edge_count} 条新边")

    # 插入新POI
    for poi in EXTRA_POI:
        cursor.execute("""
            INSERT INTO poi_points (id, name, poi_type, category, description, geom, lat, lon)
            VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (poi["id"], poi["name"], poi["type"], poi["category"], poi["description"],
              poi["lon"], poi["lat"], poi["lat"], poi["lon"]))

    conn.commit()
    print(f"[DB] 插入 {len(EXTRA_POI)} 个新POI")

    # 验证
    cursor.execute("SELECT COUNT(*) FROM road_nodes;")
    print(f"[DB] 总节点数: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM road_edges;")
    print(f"[DB] 总边数: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM poi_points;")
    print(f"[DB] 总POI数: {cursor.fetchone()[0]}")

    conn.close()
    print("[DB] 路网扩充完成！")


if __name__ == "__main__":
    main()
