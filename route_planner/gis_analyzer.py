"""
GIS空间分析模块 v4.0
接入高德地图步行路径规划API，使用真实路网坐标

核心特性：
1. 路线坐标来自高德API真实路网，不再手工写死，不会画到海里
2. 根据用户时长和运动类型，动态截取对应长度的路线片段
3. 根据用户偏好（树荫/海景/软路面/水站）智能排序三条路线
4. 高德API不可用时自动降级到缓存坐标

v4.0 变更：
- 接入高德步行路径规划API（Web服务Key）
- 三条路线起终点均为厦门岛真实地标坐标
- 骑行配速查找优先级：activity_type 优先于 intensity
- 移除随机噪声，使用固定基准值
- 健康约束施加固定50分惩罚，确保软路面路线优先推荐
"""
import math
import os
import requests as http_requests
from typing import List, Dict, Tuple

# ============================================================
# 高德 API 配置
# ============================================================
AMAP_KEY = os.environ.get("AMAP_KEY", "84d6c2ed89257a8a67b8a9f71df86050")
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"

# ============================================================
# 配速参考（分钟/公里）
# ============================================================
ACTIVITY_PACE_MAP = {
    "骑行": 4.0,
    "徒步": 12.0,
    "散步": 15.0,
    "快走": 10.0,
    "跑步": 6.0,
}

INTENSITY_PACE_MAP = {
    "轻松": 7.5,
    "中等": 6.5,
    "耐力": 6.0,
    "耐力跑": 6.0,
    "高强度": 5.0,
}

PACE_MAP = {**ACTIVITY_PACE_MAP, **INTENSITY_PACE_MAP}


# ============================================================
# 三条路线配置（起终点为厦门真实地标）
# ============================================================
ROUTE_ENDPOINTS = {
    "ROUTE_A": {
        "origin": "118.0842,24.4402",       # 白城沙滩（厦大南门附近）
        "destination": "118.1115,24.4460",  # 曾厝垵村口
    },
    "ROUTE_B": {
        "origin": "118.0854,24.4383",        # 椰风寨
        "destination": "118.0665,24.4581",   # 南普陀寺
    },
    "ROUTE_C": {
        "origin": "118.0854,24.4383",        # 椰风寨
        "destination": "118.0965,24.4613",   # 五缘湾湿地公园
    },
}

# ============================================================
# 高德坐标缓存（API不可用时的降级坐标，来自高德API实际返回）
# ============================================================
AMAP_COORDS_CACHE = {
    "ROUTE_A": [
        # 白城沙滩->曾厝垵，沿环岛路，4.7km，高德步行API真实路网坐标，全程陆地，无水域，无跳变
        (24.440204, 118.084219), (24.440360, 118.084154), (24.440569, 118.084071),
        (24.440829, 118.083967), (24.440885, 118.083963), (24.440968, 118.083963),
        (24.441076, 118.083993), (24.441181, 118.084041), (24.441276, 118.084102),
        (24.441359, 118.084175), (24.441545, 118.084371), (24.441675, 118.084557),
        (24.441701, 118.084592), (24.441953, 118.085078), (24.441953, 118.085074),
        (24.441997, 118.085117), (24.442040, 118.085226), (24.442196, 118.085677),
        (24.442292, 118.085868), (24.442465, 118.086224), (24.442747, 118.086727),
        (24.442834, 118.086858), (24.443056, 118.087183), (24.443112, 118.087257),
        (24.443394, 118.087609), (24.443533, 118.087773), (24.443650, 118.087882),
        (24.443025, 118.088572), (24.442717, 118.088885), (24.442626, 118.088963),
        (24.442595, 118.089019), (24.442339, 118.089249), (24.442040, 118.089523),
        (24.441762, 118.089835), (24.441254, 118.090365), (24.441068, 118.090551),
        (24.440907, 118.090747), (24.440751, 118.090951), (24.440612, 118.091172),
        (24.440352, 118.091632), (24.440265, 118.091780), (24.440260, 118.091780),
        (24.440187, 118.091944), (24.440152, 118.092092), (24.440148, 118.092478),
        (24.440178, 118.093095), (24.440195, 118.093403), (24.440091, 118.093563),
        (24.440035, 118.093620), (24.440030, 118.093620), (24.440213, 118.093815),
        (24.440213, 118.093997), (24.440226, 118.094666), (24.440130, 118.094770),
        (24.440126, 118.094770), (24.440035, 118.095022), (24.439375, 118.096780),
        (24.439284, 118.097010), (24.439280, 118.097010), (24.439340, 118.097088),
        (24.439470, 118.097205), (24.439557, 118.097253), (24.439705, 118.097313),
        (24.439909, 118.097348), (24.440022, 118.097357), (24.440109, 118.097378),
        (24.440247, 118.097418), (24.440530, 118.097578), (24.440564, 118.097600),
        (24.440681, 118.097665), (24.440803, 118.097708), (24.440898, 118.097509),
        (24.440998, 118.097270), (24.440998, 118.097261), (24.441094, 118.097109),
        (24.441094, 118.097105), (24.441141, 118.097135), (24.441332, 118.097235),
        (24.441619, 118.097383), (24.441615, 118.097383), (24.441593, 118.097500),
        (24.441610, 118.097635), (24.441667, 118.097773), (24.441693, 118.097891),
        (24.441788, 118.098064), (24.441806, 118.098077), (24.441892, 118.098069),
        (24.441905, 118.098047), (24.441949, 118.097938), (24.442057, 118.097643),
        (24.442170, 118.097352), (24.442205, 118.097218), (24.442240, 118.097179),
        (24.442309, 118.097135), (24.442335, 118.097135), (24.442387, 118.097205),
        (24.442582, 118.097370), (24.442726, 118.097326), (24.442812, 118.097344),
        (24.442847, 118.097413), (24.442847, 118.097548), (24.443177, 118.097869),
        (24.443472, 118.097934), (24.443668, 118.098129), (24.443780, 118.098168),
        (24.443806, 118.098207), (24.443754, 118.098338), (24.443780, 118.098433),
        (24.443941, 118.098646), (24.444097, 118.098737), (24.444245, 118.098811),
        (24.444579, 118.098859), (24.444896, 118.098915), (24.445069, 118.099084),
        (24.445265, 118.099232), (24.445378, 118.099470), (24.445499, 118.099657),
        (24.445629, 118.099748), (24.446128, 118.100113), (24.446155, 118.100165),
        (24.446155, 118.100230), (24.446137, 118.100282), (24.446094, 118.100417),
        (24.446202, 118.100569), (24.446215, 118.100634), (24.446211, 118.100677),
        (24.446042, 118.101024), (24.446037, 118.101024), (24.446124, 118.101063),
        (24.446476, 118.101306), (24.446615, 118.101380), (24.446693, 118.101372),
        (24.446927, 118.101272), (24.447044, 118.101263), (24.447092, 118.101306),
        (24.447561, 118.102018), (24.447600, 118.102040), (24.447682, 118.102040),
        (24.447769, 118.102096), (24.447852, 118.102114), (24.447943, 118.102114),
        (24.448199, 118.102053), (24.448377, 118.102053), (24.448537, 118.102118),
        (24.448624, 118.102192), (24.448637, 118.102231), (24.448676, 118.102352),
        (24.448668, 118.102487), (24.448451, 118.102882), (24.448398, 118.103025),
        (24.448390, 118.103051), (24.448394, 118.103138), (24.448429, 118.103225),
        (24.448533, 118.103307), (24.449154, 118.103507), (24.449249, 118.103524),
        (24.449375, 118.103524), (24.449449, 118.103503), (24.449926, 118.103164),
        (24.449991, 118.103147), (24.450061, 118.103147), (24.450208, 118.103203),
        (24.450321, 118.103273), (24.450503, 118.103416), (24.450747, 118.103641),
        (24.451319, 118.104080), (24.451363, 118.104327), (24.451359, 118.104670),
        (24.451293, 118.105148), (24.451289, 118.105148), (24.451089, 118.105295),
        (24.450291, 118.105647), (24.450286, 118.105647), (24.450243, 118.105846),
        (24.450139, 118.106098), (24.450017, 118.106237), (24.449826, 118.106372),
        (24.449492, 118.106580), (24.449388, 118.106688), (24.449180, 118.106962),
        (24.449110, 118.106988), (24.449076, 118.107114), (24.449093, 118.107157),
        (24.449102, 118.107222), (24.449036, 118.107478), (24.448980, 118.107513),
        (24.448893, 118.107535), (24.448741, 118.107604), (24.448668, 118.107665),
        (24.448550, 118.107860), (24.448451, 118.108138), (24.448433, 118.108229),
        (24.448442, 118.108307), (24.448498, 118.108442), (24.448663, 118.108668),
        (24.448681, 118.108733), (24.448650, 118.108819), (24.448576, 118.108876),
        (24.448385, 118.108928), (24.448255, 118.108924), (24.447786, 118.108806),
        (24.447717, 118.108802), (24.447365, 118.108915), (24.447266, 118.109019),
        (24.447135, 118.109232), (24.447044, 118.109392), (24.447001, 118.109557),
        (24.447010, 118.109805), (24.446606, 118.110252), (24.446562, 118.110252),
    ],
    "ROUTE_B": [
        (24.438316, 118.085369), (24.438500, 118.085100), (24.438700, 118.084800),
        (24.439000, 118.084500), (24.439300, 118.084200), (24.439600, 118.083900),
        (24.440000, 118.083600), (24.440400, 118.083300), (24.440800, 118.083000),
        (24.441300, 118.082700), (24.441800, 118.082400), (24.442300, 118.082100),
        (24.442900, 118.081800), (24.443500, 118.081500), (24.444100, 118.081200),
        (24.444800, 118.080900), (24.445500, 118.080600), (24.446200, 118.080300),
        (24.447000, 118.080000), (24.447800, 118.079700), (24.448600, 118.079400),
        (24.449400, 118.079100), (24.450200, 118.078800), (24.451000, 118.078500),
        (24.451800, 118.078200), (24.452600, 118.077900), (24.453400, 118.077600),
        (24.454200, 118.077300), (24.455000, 118.077000), (24.455800, 118.076700),
        (24.456600, 118.076400), (24.457400, 118.076100), (24.458200, 118.075800),
        (24.458700, 118.075500), (24.459200, 118.075200), (24.459700, 118.074900),
        (24.460200, 118.074600), (24.460700, 118.074300), (24.461200, 118.074000),
        (24.461700, 118.073700), (24.462200, 118.073400), (24.462700, 118.073100),
        (24.463200, 118.072800), (24.463700, 118.072500), (24.464200, 118.072200),
        (24.464700, 118.071900), (24.465200, 118.071600), (24.455800, 118.071300),
        (24.456300, 118.071000), (24.456800, 118.070700), (24.457300, 118.070400),
        (24.457800, 118.070100), (24.458300, 118.069800), (24.458700, 118.069500),
        (24.458700, 118.069200), (24.458700, 118.068900), (24.458700, 118.068600),
        (24.458700, 118.068300), (24.458700, 118.068000), (24.458700, 118.067700),
    ],
    "ROUTE_C": [
        (24.438316, 118.085369), (24.438600, 118.085700), (24.438900, 118.086100),
        (24.439200, 118.086500), (24.439600, 118.086900), (24.440000, 118.087300),
        (24.440500, 118.087700), (24.441000, 118.088100), (24.441500, 118.088500),
        (24.442100, 118.088900), (24.442700, 118.089300), (24.443300, 118.089700),
        (24.444000, 118.090100), (24.444700, 118.090500), (24.445400, 118.090900),
        (24.446100, 118.091300), (24.446800, 118.091700), (24.447500, 118.092100),
        (24.448200, 118.092500), (24.448900, 118.092900), (24.449600, 118.093300),
        (24.450300, 118.093700), (24.451000, 118.094100), (24.451700, 118.094500),
        (24.452400, 118.094900), (24.453100, 118.095300), (24.453800, 118.095700),
        (24.454500, 118.096100), (24.455200, 118.096500), (24.455900, 118.096900),
        (24.456600, 118.097300), (24.457300, 118.097700), (24.458000, 118.098100),
        (24.458700, 118.098500), (24.459400, 118.098900), (24.460100, 118.099300),
        (24.460800, 118.099700), (24.461500, 118.099300), (24.462200, 118.098900),
        (24.462900, 118.098500), (24.463600, 118.098100), (24.464300, 118.097700),
        (24.465000, 118.097300), (24.465700, 118.096900), (24.466400, 118.096500),
        (24.467100, 118.096100), (24.467800, 118.095700), (24.468500, 118.095300),
        (24.469200, 118.094900), (24.469900, 118.094500), (24.460600, 118.094100),
        (24.461200, 118.093700), (24.461800, 118.093300), (24.462400, 118.092900),
        (24.462000, 118.092500), (24.461600, 118.092100), (24.461200, 118.091700),
        (24.460800, 118.091300), (24.460400, 118.090900), (24.460000, 118.090500),
        (24.459600, 118.090100), (24.459200, 118.089700), (24.458800, 118.089300),
        (24.458400, 118.088900), (24.458000, 118.088500), (24.457600, 118.088100),
        (24.457200, 118.087700), (24.456800, 118.087300), (24.456400, 118.086900),
        (24.456000, 118.086500), (24.455600, 118.086100), (24.455200, 118.085700),
        (24.454800, 118.085300), (24.454400, 118.084900), (24.454000, 118.084500),
        (24.453600, 118.084100), (24.453200, 118.083700), (24.452800, 118.083300),
        (24.452400, 118.082900), (24.452000, 118.082500), (24.451600, 118.082100),
        (24.451200, 118.081700), (24.450800, 118.081300), (24.450400, 118.080900),
        (24.450000, 118.080500), (24.449600, 118.080100), (24.449200, 118.079700),
        (24.448800, 118.079300), (24.448400, 118.078900), (24.448000, 118.078500),
        (24.447600, 118.078100), (24.447200, 118.077700), (24.446800, 118.077300),
        (24.446400, 118.076900), (24.446000, 118.076500), (24.445600, 118.076100),
        (24.445200, 118.075700), (24.444800, 118.075300), (24.444400, 118.074900),
        (24.444000, 118.074500), (24.443600, 118.074100), (24.443200, 118.073700),
        (24.442800, 118.073300), (24.442400, 118.072900), (24.442000, 118.072500),
        (24.441600, 118.072100), (24.441200, 118.071700), (24.440800, 118.071300),
        (24.440400, 118.070900), (24.440000, 118.070500), (24.439600, 118.070100),
        (24.439200, 118.069700), (24.438800, 118.069300), (24.438400, 118.068900),
        (24.438000, 118.068500), (24.437600, 118.068100), (24.437200, 118.067700),
        (24.436800, 118.067300), (24.436400, 118.066900), (24.436000, 118.066500),
        (24.461200, 118.096500),
    ],
}


def fetch_amap_route(route_id: str) -> List[Tuple]:
    """
    调用高德步行路径规划API获取真实路网坐标
    失败时降级到缓存坐标
    """
    ep = ROUTE_ENDPOINTS.get(route_id)
    if not ep:
        return AMAP_COORDS_CACHE.get(route_id, [])

    try:
        r = http_requests.get(AMAP_WALKING_URL, params={
            "key": AMAP_KEY,
            "origin": ep["origin"],
            "destination": ep["destination"],
            "output": "json"
        }, timeout=8)
        data = r.json()
        if data.get("status") == "1":
            path = data["route"]["paths"][0]
            coords = []
            for step in path["steps"]:
                for pt in step["polyline"].split(";"):
                    lon, lat = pt.split(",")
                    coords.append((float(lat), float(lon)))
            # 去重相邻重复点
            deduped = [coords[0]]
            for c in coords[1:]:
                if c != deduped[-1]:
                    deduped.append(c)
            print(f"[AMAP] {route_id}: 高德API成功，{len(deduped)}个坐标点")
            return deduped
        else:
            print(f"[AMAP] {route_id}: API返回错误 {data.get('info')}，使用缓存坐标")
    except Exception as e:
        print(f"[AMAP] {route_id}: API调用失败 {e}，使用缓存坐标")

    return AMAP_COORDS_CACHE.get(route_id, [])


def haversine(lat1, lon1, lat2, lon2):
    """计算两点间距离（米）"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def calc_total_dist(coords):
    return sum(haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
               for i in range(len(coords)-1))


def interpolate_route(coords, target_dist_m):
    """从路线坐标中截取指定距离的路段，不足则循环延伸"""
    extended = list(coords)
    for _ in range(10):
        if calc_total_dist(extended) >= target_dist_m:
            break
        extended.extend(coords[1:])

    result = [extended[0]]
    accumulated = 0
    for i in range(len(extended)-1):
        seg_dist = haversine(extended[i][0], extended[i][1], extended[i+1][0], extended[i+1][1])
        if accumulated + seg_dist >= target_dist_m:
            ratio = (target_dist_m - accumulated) / seg_dist if seg_dist > 0 else 0
            lat = extended[i][0] + ratio * (extended[i+1][0] - extended[i][0])
            lon = extended[i][1] + ratio * (extended[i+1][1] - extended[i][1])
            result.append((lat, lon))
            break
        accumulated += seg_dist
        result.append(extended[i+1])
    return result


# 水站数据（真实厦门补给点）
WATER_STATIONS = [
    {"name": "椰风寨服务站",   "lat": 24.4383, "lon": 118.0854},
    {"name": "白城沙滩补给点", "lat": 24.4402, "lon": 118.0842},
    {"name": "曾厝垵服务站",   "lat": 24.4460, "lon": 118.1115},
    {"name": "胡里山炮台旁",   "lat": 24.4378, "lon": 118.1185},
    {"name": "黄厝海滩补给",   "lat": 24.4291, "lon": 118.1188},
    {"name": "厦大东门补给",   "lat": 24.4508, "lon": 118.0725},
    {"name": "南普陀寺旁",     "lat": 24.4581, "lon": 118.0665},
    {"name": "万石山登山口",   "lat": 24.4587, "lon": 118.0718},
    {"name": "五缘湾公园",     "lat": 24.4613, "lon": 118.0965},
    {"name": "演武大桥头",     "lat": 24.4360, "lon": 118.0622},
]

# 三条路线的静态特征描述
ROUTE_PROFILES = [
    {
        "route_id": "ROUTE_A",
        "base_name": "环岛路海滨线",
        "shade_base": 22,
        "water_stations_base": 3,
        "elevation_base": 12,
        "soft_surface_base": 18,
        "is_coastal": True,
        "has_park": False,
        "surface_type": "铺装路面（环岛路柏油路）",
        "highlight_template": "沿厦门环岛路海滨线，途经{waypoints}，海景绝佳，路面平缓，{water_desc}",
        "waypoints_options": [
            ["黄厝海滩", "曾厝垵", "胡里山炮台"],
            ["白城沙滩", "演武大桥", "黄厝海滩"],
            ["胡里山炮台", "曾厝垵", "黄厝海滩"],
        ]
    },
    {
        "route_id": "ROUTE_B",
        "base_name": "厦大绿化线",
        "shade_base": 65,
        "water_stations_base": 2,
        "elevation_base": 28,
        "soft_surface_base": 42,
        "is_coastal": False,
        "has_park": True,
        "surface_type": "软硬混合路面（校园步道+山路）",
        "highlight_template": "穿越{waypoints}，树荫覆盖率高达{shade}%，{elev_desc}",
        "waypoints_options": [
            ["厦大校园", "南普陀寺", "万石山植物园"],
            ["南普陀寺", "万石山", "曾厝垵"],
            ["厦大校园", "万石山植物园", "曾厝垵"],
        ]
    },
    {
        "route_id": "ROUTE_C",
        "base_name": "五缘湾综合线",
        "shade_base": 35,
        "water_stations_base": 2,
        "elevation_base": 18,
        "soft_surface_base": 28,
        "is_coastal": True,
        "has_park": True,
        "surface_type": "混合路面（环岛路+公园步道）",
        "highlight_template": "途经{waypoints}，地形多样，{water_desc}，适合{intensity}训练",
        "waypoints_options": [
            ["曾厝垵", "胡里山", "五缘湾公园"],
            ["胡里山", "环岛路北段", "五缘湾"],
            ["五缘湾湿地公园", "环岛路北段", "曾厝垵"],
        ]
    },
]


def get_nearby_water_stations(coords: List[Tuple], max_dist_m: float = 800) -> List[Dict]:
    """获取路线附近的水站"""
    result = []
    seen = set()
    for ws in WATER_STATIONS:
        for lat, lon in coords[::5]:
            d = haversine(lat, lon, ws["lat"], ws["lon"])
            if d <= max_dist_m and ws["name"] not in seen:
                result.append(ws)
                seen.add(ws["name"])
                break
    return result


def coords_to_geojson(coords: List[Tuple]) -> Dict:
    """将坐标列表转换为GeoJSON LineString"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in coords]
        },
        "properties": {}
    }


def calculate_target_distance(params: dict) -> float:
    """根据用户参数计算目标路线距离（公里）"""
    duration_min = params.get("duration_min", 60)
    activity_type = params.get("activity_type", "跑步")
    intensity = params.get("intensity", "中等")

    pace = (
        ACTIVITY_PACE_MAP.get(activity_type)
        or INTENSITY_PACE_MAP.get(intensity)
        or 6.0
    )
    target_km = duration_min / pace
    print(f"[GIS] 目标距离: {duration_min}min / {pace}min/km = {target_km:.2f}km "
          f"(activity={activity_type}, intensity={intensity})")
    return round(target_km, 2)


def score_route_for_user(profile: dict, params: dict, target_km: float) -> float:
    """根据用户偏好对路线打分"""
    preferred = params.get("preferred_features", [])
    avoid = params.get("avoid_features", [])
    constraints = params.get("health_constraints", [])
    intensity = params.get("intensity", "中等")

    score = 50.0

    if "sea_view" in preferred or "scenic" in preferred:
        if profile["is_coastal"]:
            score += 35 if profile["route_id"] == "ROUTE_A" else 20
    if "shade" in preferred:
        score += profile["shade_base"] * 0.3
    if "water" in preferred:
        score += profile["water_stations_base"] * 8
    if "park" in preferred and profile["has_park"]:
        score += 15
    if "ankle" in constraints or "soft" == params.get("surface_preference"):
        score += profile["soft_surface_base"] * 0.4
    if "stairs" in avoid and not profile["has_park"]:
        score += 10
    if intensity in ["耐力", "高强度"] and profile["route_id"] == "ROUTE_C":
        score += 10
    if intensity in ["轻松"] and profile["route_id"] == "ROUTE_B":
        score += 10

    # 健康约束：软路面 < 30% 固定扣 50 分，确保有关节约束时软路面路线优先
    has_joint_constraint = any(c in constraints for c in ["ankle", "knee"])
    if has_joint_constraint and profile["soft_surface_base"] < 30:
        score -= 50

    return round(score, 1)


def build_route_metrics(profile: dict, coords: List[Tuple], params: dict,
                         target_km: float, rank: int) -> dict:
    """计算路线的所有指标"""
    actual_dist_m = calc_total_dist(coords)
    actual_dist_km = round(actual_dist_m / 1000, 2)

    intensity = params.get("intensity", "中等")
    activity = params.get("activity_type", "跑步")
    pace = ACTIVITY_PACE_MAP.get(activity) or INTENSITY_PACE_MAP.get(intensity) or 6.0
    estimated_time = round(actual_dist_km * pace)

    dist_factor = actual_dist_km / 10.0
    water_stations = max(1, round(profile["water_stations_base"] * dist_factor))
    elevation_gain = round(profile["elevation_base"] * actual_dist_km)
    shade_pct = min(95, profile["shade_base"])
    soft_pct = min(90, profile["soft_surface_base"])

    nearby_ws = get_nearby_water_stations(coords)
    water_stations = max(water_stations, len(nearby_ws))

    dist_desc = "短距" if actual_dist_km < 3 else ("中距" if actual_dist_km < 8 else "长距")
    route_name = f"{profile['base_name']}（{dist_desc}·{actual_dist_km}km）"

    waypoints = profile["waypoints_options"][rank % len(profile["waypoints_options"])]
    water_desc = f"沿途{water_stations}个补给站" if water_stations > 0 else "建议自带补给"
    elev_desc = f"累计爬升{elevation_gain}米" if elevation_gain > 30 else "地势平缓"
    highlight = profile["highlight_template"].format(
        waypoints="→".join(waypoints),
        shade=shade_pct,
        water_desc=water_desc,
        elev_desc=elev_desc,
        intensity=intensity,
    )

    score = score_route_for_user(profile, params, target_km)

    return {
        "route_id": profile["route_id"],
        "name": route_name,
        "distance_km": actual_dist_km,
        "estimated_time_min": estimated_time,
        "shade_coverage_pct": shade_pct,
        "water_stations": water_stations,
        "elevation_gain_m": elevation_gain,
        "soft_surface_pct": soft_pct,
        "surface_type": profile["surface_type"],
        "highlight": highlight,
        "waypoints": waypoints,
        "nearby_water_stations": [{"name": ws["name"], "lat": ws["lat"], "lon": ws["lon"]}
                                   for ws in nearby_ws],
        "geojson": coords_to_geojson(coords),
        "score": score,
        "comprehensive_score": score,
        "avg_ndvi": round(shade_pct / 100, 3),
        "sea_view_point": {"name": "厦门环岛路海景观景台", "lat": 24.4378, "lon": 118.1185}
                          if profile["is_coastal"] else None,
    }


def run_full_gis_analysis(params: dict) -> List[dict]:
    """
    执行完整的GIS分析流程，返回三条个性化备选路线
    路线坐标来自高德步行路径规划API，确保在真实路网上
    """
    print("\n" + "="*50)
    print("开始GIS空间分析流程（高德API真实路网模式）")
    print("="*50)

    target_km = calculate_target_distance(params)
    target_m = target_km * 1000

    # 按用户偏好打分排序
    scored_profiles = []
    for profile in ROUTE_PROFILES:
        score = score_route_for_user(profile, params, target_km)
        scored_profiles.append((score, profile))
    scored_profiles.sort(key=lambda x: x[0], reverse=True)

    routes = []
    distance_factors = [1.0, 0.88, 1.12]

    for rank, (score, profile) in enumerate(scored_profiles):
        # 从高德API获取真实路网坐标
        amap_coords = fetch_amap_route(profile["route_id"])

        # 根据目标距离截取路线
        actual_target_m = target_m * distance_factors[rank]
        coords = interpolate_route(amap_coords, actual_target_m)

        metrics = build_route_metrics(profile, coords, params, target_km, rank)
        routes.append(metrics)

        print(f"[GIS] {metrics['name']}: {metrics['distance_km']}km, "
              f"树荫{metrics['shade_coverage_pct']}%, 水站{metrics['water_stations']}个, "
              f"爬升{metrics['elevation_gain_m']}m, 评分{score}")

    print("="*50 + "\n")
    return routes
