from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from .services.cluster_engine import cluster_students
from .services.csv_processor import CSVProcessor
from .models import CSVUpload, ProcessedStudent
from .serializers import CSVUploadSerializer, ClusteringResultSerializer, ProcessedStudentSerializer
import uuid

class ClusterStudentsView(APIView):
    """Original view for clustering individual students"""
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


class CSVUploadView(APIView):
    """Handle CSV file uploads for batch processing"""
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Upload CSV file for processing"""
        if 'csv_file' not in request.FILES:
            return Response(
                {"error": "No CSV file provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_file = request.FILES['csv_file']
        
        # Validate file type
        if not csv_file.name.endswith('.csv'):
            return Response(
                {"error": "File must be a CSV"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create CSV upload record (using test user for no-auth testing)
        from django.contrib.auth.models import User
        test_user, created = User.objects.get_or_create(
            username='test_user',
            defaults={'email': 'test@example.com', 'first_name': 'Test', 'last_name': 'User'}
        )
        
        csv_upload = CSVUpload.objects.create(
            uploaded_by=test_user,
            original_filename=csv_file.name,
            csv_file=csv_file
        )
        
        serializer = CSVUploadSerializer(csv_upload)
        return Response({
            "message": "CSV uploaded successfully",
            "upload": serializer.data
        }, status=status.HTTP_201_CREATED)
    
    def get(self, request):
        """Get list of uploaded CSV files"""
        uploads = CSVUpload.objects.all().order_by('-upload_timestamp')
        serializer = CSVUploadSerializer(uploads, many=True)
        return Response(serializer.data)


class ProcessCSVView(APIView):
    """Process uploaded CSV file through clustering pipeline"""
    permission_classes = [AllowAny]
    
    def post(self, request, upload_id):
        """Start processing of uploaded CSV file"""
        try:
            # Validate UUID format
            uuid.UUID(upload_id)
        except ValueError:
            return Response(
                {"error": "Invalid upload ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_upload = get_object_or_404(
            CSVUpload, 
            id=upload_id, 
            uploaded_by=request.user
        )
        
        if csv_upload.processing_status == 'processing':
            return Response(
                {"error": "CSV is already being processed"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if csv_upload.processing_status == 'completed':
            return Response(
                {"message": "CSV has already been processed"}, 
                status=status.HTTP_200_OK
            )
        
        # Initialize processor and start processing
        processor = CSVProcessor(csv_upload)
        result = processor.process_and_cluster()
        
        if result['success']:
            return Response({
                "message": "CSV processed successfully",
                "result": result
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {"error": result['error']}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ProcessingStatusView(APIView):
    """Check processing status of uploaded CSV"""
    permission_classes = [AllowAny]
    
    def get(self, request, upload_id):
        """Get processing status"""
        try:
            uuid.UUID(upload_id)
        except ValueError:
            return Response(
                {"error": "Invalid upload ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_upload = get_object_or_404(
            CSVUpload, 
            id=upload_id, 
            uploaded_by=request.user
        )
        
        status_data = CSVProcessor.get_processing_status(upload_id)
        return Response(status_data)


class ClusteringResultsView(APIView):
    """Get clustering results for processed CSV"""
    permission_classes = [AllowAny]
    
    def get(self, request, upload_id):
        """Get clustering results"""
        try:
            uuid.UUID(upload_id)
        except ValueError:
            return Response(
                {"error": "Invalid upload ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_upload = get_object_or_404(
            CSVUpload, 
            id=upload_id, 
            uploaded_by=request.user
        )
        
        results = CSVProcessor.get_clustering_results(upload_id)
        
        if 'error' in results:
            return Response(
                {"error": results['error']}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(results)


class ClusterDetailView(APIView):
    """Get detailed information about a specific cluster"""
    permission_classes = [AllowAny]
    
    def get(self, request, upload_id, cluster_id):
        """Get students in a specific cluster"""
        try:
            uuid.UUID(upload_id)
        except ValueError:
            return Response(
                {"error": "Invalid upload ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_upload = get_object_or_404(
            CSVUpload, 
            id=upload_id, 
            uploaded_by=request.user
        )
        
        if csv_upload.processing_status != 'completed':
            return Response(
                {"error": "CSV processing not completed"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        students_in_cluster = ProcessedStudent.objects.filter(
            csv_upload=csv_upload,
            cluster=cluster_id
        )
        
        serializer = ProcessedStudentSerializer(students_in_cluster, many=True)
        return Response({
            "cluster_id": cluster_id,
            "student_count": len(students_in_cluster),
            "students": serializer.data
        })
