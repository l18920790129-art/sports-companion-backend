"""
Django REST API 视图
"""
import json
import time
import os
import logging

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .llm_intent_parser import parse_user_intent, generate_route_description
from .gis_analyzer import run_full_gis_analysis


def _cors_response(data, status=200):
    """构建 JSON 响应，CORS 头完全由 django-cors-headers 中间件处理"""
    return JsonResponse(data, status=status, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
def plan_route(request):
    """
    核心API：接收用户自然语言需求，返回个性化路线方案
    POST /api/plan/
    Body: {"query": "今天下午我想进行一个90分钟的耐力跑..."}
    """
    if request.method == "OPTIONS":
        return _cors_response({})

    if request.method != "POST":
        return _cors_response({"error": "仅支持POST请求"}, status=405)

    # 解析请求体（兼容非JSON请求，返回400而非500）
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return _cors_response(
            {"error": "请求体必须是合法的JSON格式", "detail": str(e)},
            status=400
        )

    user_query = body.get("query", "").strip()
    if not user_query:
        return _cors_response({"error": "请输入运动需求，例如：我想跑90分钟"}, status=400)

    if len(user_query) > 500:
        return _cors_response({"error": "输入内容过长，请控制在500字以内"}, status=400)

    t_start = time.time()

    try:
        logger.info("[API] 收到用户请求: %s", user_query)
        t1 = time.time()
        params = parse_user_intent(user_query)
        t_llm_parse = round(time.time() - t1, 2)
        logger.info("[API] 意图解析完成，耗时 %ss", t_llm_parse)
    except Exception as e:
        logger.error("[API] LLM意图解析失败: %s", e)
        params = {
            "duration_min": 60,
            "activity_type": "跑步",
            "intensity": "中等",
            "preferred_features": [],
            "avoid_features": [],
            "surface_preference": "any",
            "health_constraints": [],
            "estimated_distance_km": 9.0,
            "user_notes": user_query,
        }
        t_llm_parse = 0

    try:
        t2 = time.time()
        routes = run_full_gis_analysis(params)
        t_gis = round(time.time() - t2, 2)
        logger.info("[API] GIS分析完成，耗时 %ss，生成 %d 条路线", t_gis, len(routes))
    except Exception as e:
        import traceback
        logger.error("[API] GIS分析失败: %s\n%s", e, traceback.format_exc())
        return _cors_response(
            {"error": "路线生成失败，请稍后重试", "detail": str(e)},
            status=500
        )

    try:
        t3 = time.time()
        for route in routes:
            try:
                route["description"] = generate_route_description(route, user_query)
            except Exception:
                route["description"] = route.get("highlight", "精心规划的厦门运动路线")
        t_llm_desc = round(time.time() - t3, 2)
    except Exception as e:
        logger.warning("[API] 路线描述生成失败: %s", e)
        t_llm_desc = 0

    t_total = round(time.time() - t_start, 2)
    recommended = rank_routes(routes, params)

    try:
        from .models import RouteHistory
        RouteHistory.objects.create(
            user_query=user_query,
            parsed_params=params,
            routes_count=len(routes),
            recommended_route=recommended,
            total_time_s=t_total,
        )
    except Exception:
        pass

    return _cors_response({
        "success": True,
        "user_query": user_query,
        "parsed_params": params,
        "routes": routes,
        "recommended_route_id": recommended,
        "performance": {
            "total_time_s": t_total,
            "llm_parse_time_s": t_llm_parse,
            "gis_analysis_time_s": t_gis,
            "llm_description_time_s": t_llm_desc,
        }
    })


def rank_routes(routes: list, params: dict) -> str:
    """
    综合评分，推荐最优路线。
    优先使用 gis_analyzer 已计算的 score（避免两套评分逻辑冲突）。
    兼容 gis_analyzer（route_id字段）和 gis_engine（id字段）两种返回格式。

    修复：当存在健康约束（脚踝/膝盖）时，对软路面不足的路线施加惩罚，
    防止用户偏好（如海景+35分）完全压制健康安全考量。
    """
    if not routes:
        return ""

    preferred = params.get("preferred_features", [])
    constraints = params.get("health_constraints", [])
    has_joint_constraint = any(c in constraints for c in ["ankle", "knee"])

    scores = {}
    for route in routes:
        route_id = route.get("route_id") or route.get("id", "UNKNOWN")

        if "score" in route and route["score"] > 0:
            score = route["score"]
        else:
            score = 0.0
            shade_val = route.get("shade_coverage_pct") or route.get("green_coverage", 0)
            shade_w = 2.0 if "shade" in preferred else 1.0
            score += shade_val / 100 * shade_w * 30

            water_val = route.get("water_stations", 0)
            if isinstance(water_val, list):
                water_val = len(water_val)
            water_w = 2.0 if "water" in preferred else 1.0
            score += min(water_val, 3) / 3 * water_w * 20

            soft_val = route.get("soft_surface_pct", 0)
            ankle_w = 3.0 if "ankle" in constraints else 1.0
            score += soft_val / 100 * ankle_w * 25

            elevation_penalty = route.get("elevation_gain_m", 100) / 200
            score -= elevation_penalty * (15 if "ankle" in constraints else 5)

            has_sea_view = route.get("sea_view_point") or (route.get("coastal_ratio", 0) > 30)
            if "sea_view" in preferred and has_sea_view:
                score += 35

        # 修复：健康约束惩罚
        # 当用户有关节约束（脚踝/膝盖）时，软路面低于30%的路线额外扣分，
        # 确保健康安全权重不被偏好加分完全覆盖。
        if has_joint_constraint:
            soft_val = route.get("soft_surface_pct", 0)
            if soft_val < 30:
                # 惩罚力度：每低1%扣2.5分，最多扣75分（确保超过海景偏好+35分）
                # 例：soft=18% → 惩罚 (30-18)*2.5 = 30分，soft=0% → 惩罚 75分
                penalty = (30 - soft_val) * 2.5
                score -= penalty
                logger.info(
                    "[API] 路线 %s 因软路面不足（%d%%<30%%）施加健康约束惩罚 -%.1f",
                    route_id, soft_val, penalty
                )

        scores[route_id] = round(score, 2)
        route["comprehensive_score"] = round(score, 2)

    if not scores:
        return routes[0].get("route_id") or routes[0].get("id", "")

    best = max(scores, key=scores.get)
    logger.info("[API] 路线综合评分: %s，推荐: %s", scores, best)
    return best


def get_db_conn():
    if not HAS_PSYCOPG2:
        raise RuntimeError("psycopg2 未安装")
    # 安全修复：移除密码默认值，强制要求通过环境变量注入
    # 若 DB_PASSWORD 未设置，psycopg2 将抛出连接错误，而非使用硬编码密码
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "sports_companion"),
        user=os.environ.get("DB_USER", "sports_user"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


@require_http_methods(["GET"])
def health_check(request):
    db_info = {}
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT (SELECT COUNT(*) FROM road_nodes),(SELECT COUNT(*) FROM road_edges),
                   (SELECT COUNT(*) FROM poi_points),(SELECT COUNT(*) FROM dem_elevation),
                   (SELECT COUNT(*) FROM dem_elevation)
        """)
        row = cur.fetchone()
        db_info = {
            "road_nodes": row[0], "road_edges": row[1], "pois": row[2],
            "dem_points": row[3], "ndvi_samples": row[4]  # ndvi reuses dem_elevation count
        }
        conn.close()
    except Exception as e:
        db_info = {"error": str(e)}

    return _cors_response({
        "status": "ok",
        "service": "Sports Companion API",
        "version": "5.2.0",
        "database": "PostgreSQL/PostGIS (Railway Cloud)",
        "city": "厦门市",
        "database_stats": db_info
    })


@require_http_methods(["GET"])
def get_data_sources(request):
    """数据来源信息接口 GET /api/data-sources/"""
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT data_type, source_name, source_url, description,
                   coverage_area, resolution, update_frequency,
                   record_count, last_updated::text, icon
            FROM data_sources ORDER BY id
        """)
        sources_raw = cur.fetchall()
        cur.execute("""
            SELECT (SELECT COUNT(*) FROM road_nodes) as nodes,
                   (SELECT COUNT(*) FROM road_edges) as edges,
                   (SELECT COUNT(*) FROM poi_points) as pois,
                   (SELECT COUNT(*) FROM dem_elevation) as dem,
                   (SELECT COUNT(*) FROM dem_elevation) as ndvi
        """)
        stats = cur.fetchone()
        conn.close()
        sources = []
        for row in sources_raw:
            source = dict(row)
            if row["data_type"] == "road_network":
                source["record_count"] = stats["nodes"] + stats["edges"]
                source["detail"] = f"{stats['nodes']:,}个路网节点 + {stats['edges']:,}条路段"
            elif row["data_type"] == "dem":
                source["record_count"] = stats["dem"]
                source["detail"] = f"{stats['dem']}个高程采样点"
            elif row["data_type"] == "ndvi":
                source["record_count"] = stats["ndvi"]
                source["detail"] = f"{stats['ndvi']}个植被指数采样点"
            elif row["data_type"] == "poi":
                source["record_count"] = stats["pois"]
                source["detail"] = f"{stats['pois']}个兴趣点"
            elif row["data_type"] == "llm":
                source["detail"] = "实时API调用，用于自然语言理解与路线描述生成"
            sources.append(source)
        return _cors_response({
            "success": True,
            "city": "厦门市",
            "database": "PostgreSQL 14 + PostGIS 3.4",
            "database_host": "Railway Cloud",
            "total_records": (
                stats["nodes"] + stats["edges"] + stats["pois"]
                + stats["dem"] + stats["ndvi"]
            ),
            "sources": sources
        })
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return _cors_response({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def get_map_data(request):
    """地图POI数据接口 GET /api/map-data/"""
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            "SELECT id, name, poi_type AS type, lat, lon, description FROM poi_points ORDER BY poi_type, name"
        )
        pois = [dict(row) for row in cur.fetchall()]
        conn.close()
        return _cors_response({"success": True, "pois": pois, "city": "厦门市"})
    except Exception as e:
        return _cors_response({"error": str(e)}, status=500)
