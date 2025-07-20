from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services.cluster_engine import cluster_students

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
