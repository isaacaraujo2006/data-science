from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("manual/", views.manual_predict, name="manual"),
    path("metrics/", views.metrics, name="metrics"),  # ✅ rota da aba métricas
]
