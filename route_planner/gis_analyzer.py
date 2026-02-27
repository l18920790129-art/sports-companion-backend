"""
GIS空间分析模块（本地数据库版）
数据来源：厦门市本地GIS数据库（xiamen_gis_data.py）
不依赖OSM实时请求，无需翻墙，适合生产环境部署
"""
import math
import copy
from .xiamen_gis_data import (
    ROAD_NODES, ROAD_EDGES, POI_DATA,
    PRESET_ROUTES, NDVI_ZONES, DEM_STATS
)

NODE_MAP = {n["id"]: n for n in ROAD_NODES}


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def select_route_template(params: dict) -> str:
    features = params.get("preferred_features", [])
    if "sea_view" in features or "scenic" in features:
        return "sea_view_endurance"
    return "shade_run"


def score_route(route: dict, params: dict) -> float:
    score = 0.0
    features = params.get("preferred_features", [])
    avoid = params.get("avoid_features", [])
    constraints = params.get("health_constraints", [])
    target_km = params.get("estimated_distance_km", 10.0)

    dist_diff = abs(route["distance_km"] - target_km) / max(target_km, 1)
    score += max(0, 40 * (1 - dist_diff / 0.3))

    if "shade" in features:
        score += route["shade_coverage_pct"] * 0.3
    if "water" in features:
        score += min(route["water_stations"] * 8, 20)
    if ("sea_view" in features or "scenic" in features) and route.get("sea_view"):
        score += 20
    if "ankle" in constraints:
        if route.get("ankle_friendly"):
            score += 15
        else:
            score -= 20
        if "concrete" in route["surface_type"] and "concrete" in avoid:
            score -= 10
        score -= route["elevation_gain_m"] * 0.05

    return round(score, 1)


def build_route_geojson(route: dict) -> dict:
    coords = []
    for nid in route["nodes"]:
        if nid in NODE_MAP:
            n = NODE_MAP[nid]
            coords.append([n["lon"], n["lat"]])
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "name": route["name"],
            "distance_km": route["distance_km"],
            "estimated_time_min": route["estimated_time_min"],
            "shade_coverage_pct": route["shade_coverage_pct"],
            "water_stations": route["water_stations"],
            "elevation_gain_m": route["elevation_gain_m"],
            "surface_type": route["surface_type"],
            "highlight": route["highlight"],
            "score": route.get("score", 0),
            "recommended": route.get("recommended", False),
        }
    }


def get_nearby_pois(route: dict, poi_types: list) -> list:
    node_set = set(route["nodes"])
    result = []
    for poi in POI_DATA:
        if poi["type"] in poi_types and poi.get("near_node") in node_set:
            result.append({"name": poi["name"], "type": poi["type"], "lon": poi["lon"], "lat": poi["lat"]})
    return result


def analyze_and_generate_routes(params: dict) -> dict:
    template_key = select_route_template(params)
    routes = copy.deepcopy(PRESET_ROUTES[template_key])

    for route in routes:
        route["score"] = score_route(route, params)
    routes.sort(key=lambda r: r["score"], reverse=True)
    routes[0]["recommended"] = True
    routes[1]["recommended"] = False
    routes[2]["recommended"] = False

    features = [build_route_geojson(r) for r in routes]

    preferred = params.get("preferred_features", [])
    poi_types = []
    if "water" in preferred: poi_types.append("water")
    if "sea_view" in preferred or "scenic" in preferred: poi_types.append("scenic")
    if "park" in preferred: poi_types.append("park")
    if not poi_types: poi_types = ["water", "scenic"]

    seen_pois, poi_features = set(), []
    for route in routes:
        for poi in get_nearby_pois(route, poi_types):
            if poi["name"] not in seen_pois:
                seen_pois.add(poi["name"])
                poi_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [poi["lon"], poi["lat"]]},
                    "properties": {"name": poi["name"], "type": poi["type"]}
                })

    return {
        "routes": routes,
        "geojson": {"type": "FeatureCollection", "features": features},
        "poi_geojson": {"type": "FeatureCollection", "features": poi_features},
        "ndvi_summary": {k: {"ndvi": v["ndvi_mean"], "shade_pct": v["shade_pct"]} for k, v in NDVI_ZONES.items()},
        "dem_summary": {k: {"slope_avg": v["slope_avg"], "ankle_risk": v["ankle_risk"]} for k, v in DEM_STATS.items()},
        "study_area": {"name": "厦门岛及环岛路研究区", "center": [118.0750, 24.4490], "bbox": [118.0450, 24.4340, 118.1000, 24.4650]}
    }
