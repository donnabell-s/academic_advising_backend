# clustering/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ClusterStudentsView, AdvisorViewSet, ClusterViewSet, StudentViewSet, available_clusters

router = DefaultRouter()
router.register(r'advisors', AdvisorViewSet)
router.register(r'clusters', ClusterViewSet)
router.register(r'students', StudentViewSet)

urlpatterns = [
    path('cluster-students/', ClusterStudentsView.as_view(), name='cluster-students'),
    path('', include(router.urls)),
    path('clusters/available/', available_clusters),
]
