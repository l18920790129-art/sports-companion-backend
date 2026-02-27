import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class RoutePlannerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'route_planner'

    def ready(self):
        """Django启动时预加载路网图到内存，避免第一次请求超时"""
        import threading

        def preload():
            try:
                logger.info("[Startup] 开始预加载厦门路网图...")
                from .gis_engine import load_graph_from_db
                G, nodes = load_graph_from_db()
                logger.info(
                    "[Startup] 路网图预加载完成: %d节点, %d路段",
                    G.number_of_nodes(), G.number_of_edges()
                )
            except Exception as e:
                logger.error("[Startup] 路网图预加载失败: %s", e)

        # 在后台线程中预加载，不阻塞Django启动
        t = threading.Thread(target=preload, daemon=True)
        t.start()
