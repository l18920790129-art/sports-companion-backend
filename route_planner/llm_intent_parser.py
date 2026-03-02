"""
LLM意图解析模块
将用户自然语言运动需求转化为结构化GIS查询参数
使用 OpenAI 兼容接口（支持 gemini-2.5-flash）
"""
import os
import json
import re
from openai import OpenAI

# 延迟初始化：每次调用时读取最新环境变量，避免启动时读取旧值
def get_client():
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.deepseek.com/v1')
    return OpenAI(api_key=api_key, base_url=base_url)

# 从环境变量读取模型名，默认使用 deepseek-chat
def get_model():
    return os.environ.get('LLM_MODEL', 'deepseek-chat')

INTENT_PARSE_PROMPT = """你是一个专业的运动路线规划助手，负责将用户的自然语言运动需求解析为结构化的JSON参数。

请从用户输入中提取以下信息，并以严格的JSON格式返回，字段说明如下：
duration_min: 整数，运动时长（分钟），无法判断时默认60
activity_type: 跑步/骑行/徒步/散步
intensity: 轻松/中等/耐力/高强度
preferred_features: 列表，可包含 shade(树荫)/water(水站)/scenic(风景)/sea_view(海景)/park(公园)
avoid_features: 列表，可包含 stairs(台阶)/concrete(水泥路)/traffic(车流)
surface_preference: soft(软地面)/hard(硬地面)/any(均可)
health_constraints: 列表，可包含 ankle(脚踝不适)/knee(膝盖不适)/heart(心脏问题)
estimated_distance_km: 浮点数，根据配速和时长估算路线距离
user_notes: 其他备注

配速参考（分钟/公里）：
- 轻松跑: 7.5 min/km
- 中等跑: 6.5 min/km
- 耐力跑: 6.0 min/km
- 高强度跑: 5.0 min/km
- 散步: 15.0 min/km
- 徒步: 12.0 min/km
- 骑行: 3.0 min/km

规则：
- estimated_distance_km必须根据 duration_min 和对应配速计算：estimated_distance_km = duration_min / 配速
- 若用户提到脚踝不适，surface_preference设为soft，avoid_features加入stairs和concrete
- 若用户提到耐力跑，intensity设为耐力，estimated_distance_km = duration_min / 6.0
- 若用户提到散步，activity_type设为散步，intensity设为轻松，estimated_distance_km = duration_min / 15.0
- 只返回JSON，不要有任何多余文字

用户输入：{user_input}
"""


def parse_user_intent(user_input: str) -> dict:
    """
    调用LLM解析用户自然语言意图，返回结构化参数字典
    """
    prompt = INTENT_PARSE_PROMPT.format(user_input=user_input)
    client = get_client()

    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": "你是专业的运动路线规划助手，只返回JSON格式数据。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    # 清理可能的markdown代码块
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = re.sub(r'^```\s*', '', raw)

    parsed = json.loads(raw)
    return parsed


def generate_route_description(route: dict, user_input: str) -> str:
    """
    调用LLM为路线生成人性化的语音介绍文本
    """
    prompt = f"""你是一个热情的运动教练，请根据以下路线数据，用中文生成一段简洁、生动、有感染力的路线推荐语（100字以内）。

用户需求：{user_input}

路线数据：
- 路线名称：{route['name']}
- 总距离：{route['distance_km']:.1f} 公里
- 预计用时：{route['estimated_time_min']} 分钟
- 树荫覆盖率：{route['shade_coverage_pct']}%
- 沿途水站数量：{route['water_stations']} 个
- 累计爬升：{route['elevation_gain_m']} 米
- 路面类型：{route['surface_type']}
- 特色亮点：{route['highlight']}

请直接给出推荐语，不要有"推荐语："等前缀。"""

    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
