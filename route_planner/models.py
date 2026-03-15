"""
Django 数据模型
- RouteHistory: 路线规划历史记录
- UserMemory: 用户长期记忆（偏好统计、查询历史），数据库持久化
"""
from django.db import models


class RouteHistory(models.Model):
    """路线规划历史记录"""
    user_query = models.TextField(verbose_name="用户查询")
    parsed_params = models.JSONField(default=dict, verbose_name="解析参数")
    routes_count = models.IntegerField(default=0, verbose_name="路线数量")
    recommended_route = models.CharField(max_length=50, blank=True, verbose_name="推荐路线")
    total_time_s = models.FloatField(default=0.0, verbose_name="总耗时(秒)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "route_history"
        ordering = ["-created_at"]
        verbose_name = "路线历史"
        verbose_name_plural = "路线历史"

    def __str__(self):
        return f"{self.user_query[:50]} -> {self.recommended_route}"


class UserMemory(models.Model):
    """
    用户长期记忆系统（PostgreSQL数据库持久化）
    存储用户偏好统计、查询历史、路线反馈
    使用 session_id 区分不同用户（无登录系统时用IP或固定标识）
    """
    session_id = models.CharField(
        max_length=100, default="default", db_index=True, verbose_name="会话ID"
    )
    session_count = models.IntegerField(default=0, verbose_name="会话次数")
    preference_stats = models.JSONField(default=dict, verbose_name="偏好统计")
    route_feedback = models.JSONField(default=dict, verbose_name="路线反馈")
    query_history = models.JSONField(default=list, verbose_name="查询历史(最近50条)")
    last_activity_type = models.CharField(max_length=50, blank=True, verbose_name="最近活动类型")
    last_updated = models.DateTimeField(auto_now=True, verbose_name="最后更新时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "user_memory"
        verbose_name = "用户记忆"
        verbose_name_plural = "用户记忆"

    def __str__(self):
        return f"UserMemory[{self.session_id}] sessions={self.session_count}"

    def add_query(self, user_query: str, params: dict, recommended_route: str):
        """添加一条查询记录并更新偏好统计"""
        from datetime import datetime
        self.session_count += 1

        # 记录查询历史（最多50条）
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
        history = self.query_history if isinstance(self.query_history, list) else []
        history.insert(0, entry)
        self.query_history = history[:50]

        # 更新偏好统计
        stats = self.preference_stats if isinstance(self.preference_stats, dict) else {}
        for feature in params.get("preferred_features", []):
            stats[feature] = stats.get(feature, 0) + 1
        for constraint in params.get("health_constraints", []):
            key = f"constraint_{constraint}"
            stats[key] = stats.get(key, 0) + 1
        self.preference_stats = stats
        self.last_activity_type = params.get("activity_type", "跑步")
        self.save()

    def get_context_string(self, params: dict) -> str:
        """生成记忆上下文字符串，用于RAG增强"""
        if self.session_count == 0:
            return ""
        context_parts = []
        stats = self.preference_stats if isinstance(self.preference_stats, dict) else {}
        if stats:
            top_prefs = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
            pref_names = {
                "sea_view": "海景", "shade": "树荫", "water": "水站",
                "park": "公园", "scenic": "风景",
                "constraint_ankle": "脚踝保护", "constraint_knee": "膝盖保护"
            }
            top_str = "、".join([pref_names.get(k, k) for k, _ in top_prefs])
            context_parts.append(f"用户历史偏好：{top_str}")
        history = self.query_history if isinstance(self.query_history, list) else []
        recent = history[:3]
        if recent:
            activities = [r.get("activity_type", "跑步") for r in recent]
            most_common = max(set(activities), key=activities.count)
            context_parts.append(f"最近常做运动：{most_common}")
            context_parts.append(f"累计使用{self.session_count}次")
        return "；".join(context_parts) if context_parts else ""
