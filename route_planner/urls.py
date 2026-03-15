from django.urls import path
from . import views

urlpatterns = [
    path('plan/', views.plan_route, name='plan_route'),
    path('health/', views.health_check, name='health_check'),
    path('map-data/', views.get_map_data, name='get_map_data'),
    path('data-sources/', views.get_data_sources, name='get_data_sources'),
    # v2.0 新增接口
    path('knowledge-graph/', views.get_knowledge_graph, name='get_knowledge_graph'),
    path('memory/', views.get_memory, name='get_memory'),
    path('rag-knowledge/', views.get_rag_knowledge, name='get_rag_knowledge'),
    path('water-stations/', views.get_water_stations, name='get_water_stations'),
]
