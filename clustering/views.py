from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import get_object_or_404
from rest_framework import viewsets
from .services.cluster_engine import cluster_students
from .services.csv_processor import CSVProcessor
from .models import CSVUpload, ProcessedStudent, Student, Advisor, Cluster
from .serializers import CSVUploadSerializer, ClusteringResultSerializer, ProcessedStudentSerializer, StudentSerializer, AdvisorSerializer, ClusterSerializer
from django.db.models import Count
import uuid
import re # Import regex for program/year extraction

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
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('csv_file')
        if not file_obj:
            return Response({"error": "No CSV file provided."}, status=status.HTTP_400_BAD_REQUEST)

        csv_upload = CSVUpload.objects.create(
            uploaded_by=request.user if request.user.is_authenticated else None,
            original_filename=file_obj.name,
            csv_file=file_obj,
            processing_status='uploaded'
        )

        processor = CSVProcessor(csv_upload)
        processing_result = processor.process_and_cluster()

        if processing_result['success']:
            serializer = ClusteringResultSerializer(processing_result)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(
                {"error": processing_result.get('error', 'An unknown error occurred during processing.')},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ClusterStatusView(APIView):
    """Retrieve the status of a CSV upload and processing results."""
    permission_classes = [AllowAny]

    def get(self, request, upload_id, format=None):
        csv_upload = get_object_or_404(CSVUpload, id=upload_id)
        
        if csv_upload.processing_status != 'completed':
            return Response(
                {"error": "CSV processing not completed", "status": csv_upload.processing_status},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = CSVProcessor.get_clustering_results(upload_id)
        if results.get('error'):
            return Response({"error": results['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = ClusteringResultSerializer(results)
        return Response(serializer.data)

class StudentsInClusterView(APIView):
    """Retrieve students belonging to a specific cluster for a given CSV upload."""
    permission_classes = [AllowAny]

    def get(self, request, upload_id, cluster_id, format=None):
        try:
            cluster_id = int(cluster_id)
        except ValueError:
            return Response(
                {"error": "Invalid cluster_id. Must be an integer."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_upload = get_object_or_404(CSVUpload, id=upload_id)
        
        if csv_upload.processing_status != 'completed':
            return Response(
                {"error": "CSV processing not completed"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        students_in_cluster = ProcessedStudent.objects.filter(
            csv_upload=csv_upload,
            cluster=cluster_id
        ).select_related('student')
        
        serializer = ProcessedStudentSerializer(students_in_cluster, many=True)
        return Response({
            "cluster_id": cluster_id,
            "student_count": len(students_in_cluster),
            "students": serializer.data
        })


class AdvisorViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Advisor.objects.select_related('cluster').all()
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
    clusters = Cluster.objects.all()
    serializer = ClusterSerializer(clusters, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_student_count(request):
    count = Student.objects.count()
    return Response({"student_count": count}, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_advisor_count(request):
    count = Advisor.objects.count()
    return Response({"advisor_count": count}, status=status.HTTP_200_OK)


class GraphDataView(APIView):
    """
    Endpoint to provide data for various graphs.
    Requires 'graph_type' query parameter.
    For 'student_clusters', also accepts 'secondary_feature'.
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        graph_type = request.query_params.get('graph_type')

        if not graph_type:
            return Response({"error": "Missing 'graph_type' query parameter."}, status=status.HTTP_400_BAD_REQUEST)

        data = []
        if graph_type == 'students_per_level':
            year_level_counts = Student.objects.values('program_and_grade').annotate(count=Count('program_and_grade'))
            
            processed_data = {}
            total_students = Student.objects.count()
            
            for entry in year_level_counts:
                program_grade_str = entry['program_and_grade']
                match = re.search(r'(\d+)$', program_grade_str)
                year_num = int(match.group(1)) if match else 0

                year_level_map = {
                    1: '1st Year', 2: '2nd Year', 3: '3rd Year', 4: '4th Year',
                    5: '5th Year'
                }
                year_level_label = year_level_map.get(year_num, f'Year {year_num}')

                if year_level_label not in processed_data:
                    processed_data[year_level_label] = {'students': 0, 'percentage': 0}
                
                processed_data[year_level_label]['students'] += entry['count']
            
            for label, values in processed_data.items():
                percentage = (values['students'] / total_students * 100) if total_students > 0 else 0
                data.append({
                    'yearLevel': label,
                    'students': values['students'],
                    'percentage': round(percentage, 2)
                })
            
            data.sort(key=lambda x: int(re.search(r'\d+', x['yearLevel']).group(0)) if re.search(r'\d+', x['yearLevel']) else 0)

        elif graph_type == 'students_per_program':
            program_counts = Student.objects.values('program_and_grade').annotate(count=Count('program_and_grade'))

            processed_data = {}
            total_students = Student.objects.count()

            for entry in program_counts:
                program_grade_str = entry['program_and_grade']
                match = re.match(r'([A-Za-z]+)', program_grade_str)
                program_name = match.group(1) if match else 'Unknown Program'

                if program_name not in processed_data:
                    processed_data[program_name] = {'students': 0, 'percentage': 0}
                
                processed_data[program_name]['students'] += entry['count']
            
            for name, values in processed_data.items():
                percentage = (values['students'] / total_students * 100) if total_students > 0 else 0
                data.append({
                    'program': name,
                    'students': values['students'],
                    'percentage': round(percentage, 2)
                })
            
            data.sort(key=lambda x: x['students'], reverse=True)

        elif graph_type == 'student_clusters':
            secondary_feature = request.query_params.get('secondary_feature', 'financial_status') # Default to financial_status

            # Validate secondary_feature against allowed fields in ProcessedStudent
            allowed_features = [
                'financial_status', 'workload_rating', 'help_seeking',
                'personality', 'hobby_count', 'birth_order',
                'has_external_responsibilities', 'average' # 'average' is academic performance (x-axis)
            ]
            if secondary_feature not in allowed_features:
                return Response({"error": f"Invalid secondary_feature: {secondary_feature}. Allowed: {', '.join(allowed_features)}"}, status=status.HTTP_400_BAD_REQUEST)

            # Fetch all ProcessedStudents
            all_processed_students = ProcessedStudent.objects.select_related('student').all()

            clusters_data = {}
            for student_record in all_processed_students:
                cluster_id = student_record.cluster
                if cluster_id not in clusters_data:
                    clusters_data[cluster_id] = []
                
                # X-axis: Academic Performance (average)
                academic_performance = student_record.average
                academic_performance = academic_performance if academic_performance is not None else 0

                # Y-axis: Dynamically selected secondary feature
                secondary_value = getattr(student_record, secondary_feature, 0) # Get attribute dynamically
                secondary_value = secondary_value if secondary_value is not None else 0

                clusters_data[cluster_id].append({
                    'x': round(academic_performance, 2),
                    'y': round(secondary_value, 2)
                })
            
            cluster_labels = {
                0: 'Cluster 0 (High Achievers)',
                1: 'Cluster 1 (Struggling Students)',
                2: 'Cluster 2 (Balanced Learners)',
                3: 'Cluster 3 (High Workload)'
            }
            cluster_colors = {
                0: '#10B981', # Green
                1: '#EF4444', # Red
                2: '#2563EB', # Blue
                3: '#F59E0B'  # Orange
            }

            datasets = []
            for cluster_id, points in clusters_data.items():
                datasets.append({
                    'label': cluster_labels.get(cluster_id, f'Cluster {cluster_id}'),
                    'data': points,
                    'backgroundColor': cluster_colors.get(cluster_id, '#6B7280'),
                    'pointRadius': 4,
                })
            data = {'datasets': datasets}

        else:
            return Response({"error": "Invalid 'graph_type' provided."}, status=status.HTTP_400_BAD_REQUEST)

        return Response(data, status=status.HTTP_200_OK)
