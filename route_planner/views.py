"""
Django REST API视图 v5.0
集成真实GIS引擎（NetworkX + PostgreSQL路网）和DeepSeek LLM意图解析
"""
import json
import time
import os
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .llm_intent_parser import parse_user_intent
from .gis_engine import plan_routes, DB_CONFIG


@require_http_methods(["GET"])
def health_check(request):
    """健康检查接口"""
    return JsonResponse({
        "status": "ok",
        "version": "5.0",
        "data_source": "PostgreSQL + NetworkX + 真实厦门OSM路网",
        "features": [
            "真实路网34812节点65965路段",
            "多段环路算法（三角形路线）",
            "NetworkX加权最短路径",
            "NDVI植被分析",
            "目标距离误差<15%",
        ],
    })


@csrf_exempt
@require_http_methods(["POST"])
def plan_route(request):
    """
    路线规划主接口
    POST /api/plan/
    请求体: {"query": "今天下午我想进行一个90分钟的耐力跑..."}
    返回: 三条路线方案 + LLM推荐语
    """
    t0 = time.time()

    try:
        body = json.loads(request.body.decode("utf-8"))
        user_query = body.get("query", "").strip()

        if not user_query:
            return JsonResponse({"error": "请输入运动需求描述"}, status=400)

        # === 阶段1：LLM意图解析 ===
        t1 = time.time()
        intent = parse_user_intent(user_query)
        t_llm = round(time.time() - t1, 2)
        print(f"[API] 意图解析完成 {t_llm}s: {intent}")

        # === 阶段2：GIS路线规划（NetworkX + PostgreSQL） ===
        t2 = time.time()
        routes = plan_routes(intent)
        t_gis = round(time.time() - t2, 2)
        print(f"[API] GIS规划完成 {t_gis}s，生成 {len(routes)} 条路线")

        if not routes:
            return JsonResponse(
                {"error": "未能生成路线，请检查数据库连接", "intent": intent},
                status=500,
            )

        # === 阶段3：LLM生成路线推荐语 ===
        t3 = time.time()
        routes_with_desc = _add_route_descriptions(routes, intent, user_query)
        t_desc = round(time.time() - t3, 2)

        t_total = round(time.time() - t0, 2)

        # 构建起点信息
        start_point = {
            "lat": intent.get("start_lat", 24.4380),
            "lon": intent.get("start_lon", 118.0850),
            "name": intent.get("start_location", "椰风寨"),
        }

        # 构建POI GeoJSON（水站）
        poi_features = []
        for route in routes_with_desc:
            for ws in route.get("water_stations", []):
                poi_features.append({
                    "type": "Feature",
                    "properties": {
                        "name": ws.get("name", "水站"),
                        "type": "water",
                        "category": "补给水站",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [ws.get("lon", 118.0850), ws.get("lat", 24.4380)],
                    },
                })

        return JsonResponse(
            {
                "success": True,
                "user_query": user_query,
                "parsed_params": intent,
                "routes": routes_with_desc,
                "recommended_route_id": routes_with_desc[0]["id"] if routes_with_desc else None,
                "start_point": start_point,
                "poi_geojson": {
                    "type": "FeatureCollection",
                    "features": poi_features,
                },
                "data_source": "PostgreSQL + NetworkX + 真实厦门OSM路网（34812节点）",
                "performance": {
                    "total_time_s": t_total,
                    "llm_parse_time_s": t_llm,
                    "gis_analysis_time_s": t_gis,
                    "llm_description_time_s": t_desc,
                },
            },
            json_dumps_params={"ensure_ascii": False},
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "请求体格式错误，需要JSON"}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"服务器内部错误: {str(e)}"}, status=500)


def _add_route_descriptions(routes, intent, user_query):
    """为每条路线生成LLM自然语言推荐语，并补全前端所需字段"""
    result = []
    rank_labels = ["推荐", "备选", "参考"]

    for i, route in enumerate(routes):
        rank = rank_labels[i] if i < 3 else "参考"

        # 生成LLM推荐语
        description = _generate_description(route, intent, user_query, rank)

        route_copy = dict(route)
        route_copy["description"] = description
        route_copy["rank"] = rank

        # ===== 前端字段兼容映射 =====
        # 路线ID和名称
        route_copy["route_id"] = route.get("id", chr(65 + i))
        route_copy["label"] = route.get("name", f"路线{chr(65+i)}")

        # 距离（前端用 total_length_km）
        dist_km = route.get("distance_km", 0)
        route_copy["total_length_km"] = dist_km

        # 树荫/植被（前端用 shade_pct）
        green = route.get("green_coverage", 0)
        route_copy["shade_pct"] = green
        route_copy["shade_coverage_pct"] = green

        # 水站数量（前端用 water_stations 数字）
        ws_count = route.get("water_station_count", 0)
        route_copy["water_stations"] = ws_count

        # 海景点（前端用 sea_view_pois）
        coastal = route.get("coastal_ratio", 0)
        route_copy["sea_view_pois"] = max(1, int(coastal / 20)) if coastal > 0 else 0

        # 台阶（前端用 has_steps）
        route_copy["has_steps"] = False  # 默认无台阶（算法已避免台阶）

        # 路面类型（前端用 dominant_surface）
        route_copy["dominant_surface"] = "asphalt"

        # 综合评分
        score = min(100, int(
            dist_km * 3 +
            green * 0.3 +
            coastal * 0.2 +
            ws_count * 5
        ))
        route_copy["comprehensive_score"] = score
        route_copy["score"] = score

        # 坐标（前端用 coords，格式 [[lat,lon],...]）
        coords = route.get("coordinates", [])
        # 确保格式正确（已是 [[lat,lon],...] 格式）
        route_copy["coords"] = coords

        result.append(route_copy)

    return result


def _generate_description(route, intent, user_query, rank):
    """生成单条路线的LLM推荐语"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        )

        prompt = f"""你是专业跑步教练AI，请用热情专业的语气为以下路线写一段80-120字的推荐语。

路线数据（真实厦门OSM路网）：
- 名称：{route['name']}
- 距离：{route['distance_km']}公里
- 累计爬升：{route['elevation_gain_m']}米
- 植被覆盖率：{route['green_coverage']}%（NDVI分析）
- 水站数量：{route['water_station_count']}个
- 海景比例：{route['coastal_ratio']}%
- 预计用时：{route['estimated_time_min']}分钟
- 难度：{route['difficulty']}

用户需求：{user_query}
特殊情况：{'左脚踝不适，需避免台阶和硬路面' if intent.get('ankle_issue') else '无'}

直接输出推荐语，不要有任何前缀或引号。"""

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"[API] LLM描述生成失败: {e}")
        return (
            f"{rank}路线！{route['name']}全程{route['distance_km']}公里，"
            f"植被覆盖{route['green_coverage']}%，途经{route['water_station_count']}个水站，"
            f"累计爬升{route['elevation_gain_m']}米，预计用时{route['estimated_time_min']}分钟。"
        )


@require_http_methods(["GET"])
def get_map_data(request):
    """获取地图底层数据（高NDVI路段用于前端热力图）"""
    import psycopg2

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT e.source, e.target, e.ndvi,
                   n1.lat AS lat1, n1.lon AS lon1,
                   n2.lat AS lat2, n2.lon AS lon2
            FROM road_edges e
            JOIN road_nodes n1 ON e.source = n1.id
            JOIN road_nodes n2 ON e.target = n2.id
            WHERE e.ndvi > 0.40
            LIMIT 2000
        """)

        features = []
        for row in cur.fetchall():
            _, _, ndvi, lat1, lon1, lat2, lon2 = row
            features.append({
                "type": "Feature",
                "properties": {"ndvi": float(ndvi or 0)},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]],
                },
            })

        cur.close()
        conn.close()

        return JsonResponse({
            "type": "FeatureCollection",
            "features": features,
            "total_edges": 65965,
            "high_ndvi_edges": len(features),
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
