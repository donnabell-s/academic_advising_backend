from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from .services.cluster_engine import cluster_students
from .models import Advisor, Cluster, Student
from .serializers import AdvisorSerializer, ClusterSerializer, StudentSerializer

class ClusterStudentsView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        students = request.data.get("students", [])

        if not students:
            return Response({"error": "No student data provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            clustered_data = cluster_students(students)
            return Response({"clustered": clustered_data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AdvisorViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Advisor.objects.all()
    serializer_class = AdvisorSerializer

class ClusterViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Cluster.objects.all()
    serializer_class = ClusterSerializer

class StudentViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Student.objects.all()
    serializer_class = StudentSerializer

@api_view(['GET'])
def available_clusters(request):
    permission_classes = [AllowAny]
    clusters = Cluster.objects.filter(advisor__isnull=True)
    serializer = ClusterSerializer(clusters, many=True)
    return Response(serializer.data)