"""
LLM意图解析模块 v2.0
将用户自然语言运动需求转化为结构化GIS查询参数
使用DeepSeek API（国内可用，低成本）
"""
import os
import json
import re
from openai import OpenAI

# DeepSeek API配置（通过环境变量注入，支持Railway部署）
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
)

INTENT_PARSE_PROMPT = """你是专业的运动路线规划AI助手，将用户的自然语言运动需求解析为严格的JSON格式参数。

字段说明：
- duration_min: 整数，运动时长（分钟），默认60
- sport_type: 字符串，如"耐力跑"/"慢跑"/"骑行"/"徒步"
- intensity: 字符串，"轻松"/"中等"/"耐力"/"高强度"
- need_shade: 布尔，是否需要树荫
- need_water: 布尔，是否需要水站
- need_sea_view: 布尔，是否需要海景
- ankle_issue: 布尔，是否有脚踝不适
- avoid_steps: 布尔，是否避免台阶
- avoid_hard_surface: 布尔，是否避免硬路面（水泥/石板）
- target_distance_km: 浮点数，目标距离
  - 耐力跑：duration_min / 6.0（配速6min/km）
  - 慢跑：duration_min / 7.5（配速7.5min/km）
  - 徒步：duration_min / 15.0（配速15min/km）
- pace_min_per_km: 浮点数，配速（分钟/公里）
- user_notes: 字符串，其他备注

规则：
1. 若提到脚踝/膝盖不适，ankle_issue=true，avoid_steps=true，avoid_hard_surface=true
2. 若提到耐力跑，intensity="耐力"，pace_min_per_km=6.0
3. 若提到海景/看海/海边/海，need_sea_view=true
4. 只返回JSON，不要有任何多余文字、代码块标记

用户输入：{user_input}"""


def parse_user_intent(user_input: str) -> dict:
    """调用LLM解析用户自然语言意图，返回结构化参数字典"""
    prompt = INTENT_PARSE_PROMPT.format(user_input=user_input)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业的运动路线规划助手，只返回JSON格式数据，不含任何其他文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw.strip())
    except Exception as e:
        print(f"[LLM] 意图解析失败，使用规则降级: {e}")
        parsed = _rule_based_parse(user_input)

    # 补全默认值
    parsed.setdefault("duration_min", 60)
    parsed.setdefault("sport_type", "跑步")
    parsed.setdefault("intensity", "中等")
    parsed.setdefault("need_shade", False)
    parsed.setdefault("need_water", False)
    parsed.setdefault("need_sea_view", False)
    parsed.setdefault("ankle_issue", False)
    parsed.setdefault("avoid_steps", parsed.get("ankle_issue", False))
    parsed.setdefault("avoid_hard_surface", parsed.get("ankle_issue", False))

    # 计算目标距离（核心修复：耐力跑6min/km，90min=15km）
    pace = parsed.get("pace_min_per_km")
    if not pace:
        pace_map = {"耐力": 6.0, "中等": 7.0, "轻松": 8.0, "高强度": 5.0}
        pace = pace_map.get(parsed.get("intensity", "中等"), 7.0)
        parsed["pace_min_per_km"] = pace

    if not parsed.get("target_distance_km"):
        parsed["target_distance_km"] = round(parsed["duration_min"] / pace, 1)

    return parsed


def generate_route_description(route: dict, user_input: str) -> str:
    """调用LLM为路线生成人性化的推荐语"""
    prompt = f"""你是热情的运动教练，请根据以下路线数据，用中文生成一段简洁、生动、有感染力的路线推荐语（80-120字）。

用户需求：{user_input}

路线数据：
- 路线名称：{route.get('name', '')}
- 总距离：{route.get('total_length_km', 0):.1f} 公里
- 树荫覆盖率：{route.get('shade_pct', 0)}%
- 沿途水站数量：{route.get('water_stations', 0)} 个
- 累计爬升：{route.get('elevation_gain_m', 0)} 米
- 路面类型：{route.get('dominant_surface', '混合')}
- 海景观测点：{route.get('sea_view_pois', 0)} 个
- 是否有台阶：{'有' if route.get('has_steps') else '无'}

请直接给出推荐语，不要有"推荐语："等前缀。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] 路线描述生成失败: {e}")
        shade = route.get('shade_pct', 0)
        water = route.get('water_stations', 0)
        dist = route.get('total_length_km', 0)
        return f"全程{dist:.1f}公里，树荫覆盖{shade}%，途经{water}个水站，路面平整，适合您今天的运动计划！"


def _rule_based_parse(user_input: str) -> dict:
    """基于关键词的规则解析（LLM失败时的降级方案）"""
    text = user_input
    result = {
        "duration_min": 60,
        "sport_type": "跑步",
        "intensity": "中等",
        "need_shade": False,
        "need_water": False,
        "need_sea_view": False,
        "ankle_issue": False,
        "avoid_steps": False,
        "avoid_hard_surface": False,
    }

    m = re.search(r'(\d+)\s*分钟', text)
    if m:
        result["duration_min"] = int(m.group(1))

    if "耐力" in text:
        result["sport_type"] = "耐力跑"
        result["intensity"] = "耐力"
        result["pace_min_per_km"] = 6.0
    elif "慢跑" in text:
        result["sport_type"] = "慢跑"
        result["intensity"] = "轻松"
        result["pace_min_per_km"] = 7.5

    if any(w in text for w in ["树荫", "遮阴", "阴凉"]):
        result["need_shade"] = True
    if any(w in text for w in ["水站", "补水", "饮水"]):
        result["need_water"] = True
    if any(w in text for w in ["海景", "看海", "海边", "海"]):
        result["need_sea_view"] = True
    if any(w in text for w in ["脚踝", "踝"]):
        result["ankle_issue"] = True
        result["avoid_steps"] = True
        result["avoid_hard_surface"] = True

    return result


if __name__ == "__main__":
    test_input = "今天下午我想进行一个90分钟的耐力跑，希望沿途有树荫和水站，最后能看到一段海景。我最近左脚踝有点不适。"
    print("=== 测试意图解析 ===")
    result = parse_user_intent(test_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n目标距离: {result['target_distance_km']} km（{result['duration_min']}分钟 / {result['pace_min_per_km']}min/km）")
