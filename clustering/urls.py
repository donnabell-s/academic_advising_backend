# clustering/urls.py
from django.urls import path
from .views import ClusterStudentsView

urlpatterns = [
    path('cluster-students/', ClusterStudentsView.as_view(), name='cluster-students'),
]
