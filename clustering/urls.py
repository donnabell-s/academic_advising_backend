# clustering/urls.py
from .views import (
    ClusterStudentsView, 
    CSVUploadView, 
    ProcessCSVView, 
    ProcessingStatusView, 
    ClusteringResultsView,
    ClusterDetailView
)
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ClusterStudentsView, AdvisorViewSet, ClusterViewSet, StudentViewSet, available_clusters

router = DefaultRouter()
router.register(r'advisors', AdvisorViewSet)
router.register(r'clusters', ClusterViewSet)
router.register(r'students', StudentViewSet)

urlpatterns = [
    # Original clustering endpoint
    path('cluster-students/', ClusterStudentsView.as_view(), name='cluster-students'),
    
    # CSV upload and processing endpoints
    path('upload-csv/', CSVUploadView.as_view(), name='upload-csv'),
    path('process-csv/<str:upload_id>/', ProcessCSVView.as_view(), name='process-csv'),
    path('status/<str:upload_id>/', ProcessingStatusView.as_view(), name='processing-status'),
    path('results/<str:upload_id>/', ClusteringResultsView.as_view(), name='clustering-results'),
    path('results/<str:upload_id>/cluster/<int:cluster_id>/', ClusterDetailView.as_view(), name='cluster-detail'),
    path('', include(router.urls)),
    path('clusters/available/', available_clusters),
]
