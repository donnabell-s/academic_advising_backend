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
from .models import CSVUpload, ProcessedStudent, Student, Advisor, Cluster, PCAComponent
from .serializers import CSVUploadSerializer, ClusteringResultSerializer, ProcessedStudentSerializer, StudentSerializer, AdvisorSerializer, ClusterSerializer, PCAComponentSerializer
from django.db.models import Count
import uuid
import re # Import regex for program/year extraction
from openpyxl import Workbook
from django.utils import timezone
from django.http import HttpResponse

class ExportClustersExcelView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        from openpyxl import Workbook
        from django.utils import timezone
        from django.http import HttpResponse

        # Create workbook
        wb = Workbook()

        # Fetch all clusters with related data
        clusters = Cluster.objects.select_related('advisor').prefetch_related('student').all()

        for idx, cluster in enumerate(clusters):
            # For first cluster, use the active sheet; otherwise create new
            ws = wb.active if idx == 0 else wb.create_sheet()
            ws.title = f"Cluster {cluster.cluster_id}"

            # Row 1: Cluster Name
            ws.append([f"Cluster Name: {cluster.name}"])
            # Row 2: Advisor Name
            advisor_name = cluster.advisor.name if cluster.advisor else "Unassigned"
            ws.append([f"Advisor: {advisor_name}"])

            # Empty row for spacing (optional)
            ws.append([])

            # Header for students
            ws.append(["Student ID", "Student Name", "Program & Grade"])

            students = cluster.student.all()
            if students.exists():
                for student in students:
                    ws.append([
                        student.student_id,
                        student.name or "Unnamed",
                        student.program_and_grade,
                    ])
            else:
                ws.append(["-", "No Students", "-"])

        # Prepare HTTP response with Excel MIME type
        response = HttpResponse(
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        filename = f"clusters_export_{timezone.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        response['Content-Disposition'] = f'attachment; filename={filename}'

        wb.save(response)
        return response


class PCAComponentViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = PCAComponent.objects.all()
    serializer_class = PCAComponentSerializer

class ClusterViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    queryset = Cluster.objects.annotate(student_count=Count('student'))
    serializer_class = ClusterSerializer

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
    Endpoint to provide data for various graphs:
    - students_per_level
    - students_per_program
    - student_clusters (academic performance vs selected feature)
    - pca_scatter (PCx vs PCy)
    """
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        graph_type = request.query_params.get('graph_type')

        if not graph_type:
            return Response({"error": "Missing 'graph_type' query parameter."}, status=status.HTTP_400_BAD_REQUEST)

        data = []

        # 1️⃣ Students per Level
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
                year_label = year_level_map.get(year_num, f'Year {year_num}')

                if year_label not in processed_data:
                    processed_data[year_label] = {'students': 0, 'percentage': 0}

                processed_data[year_label]['students'] += entry['count']

            for label, values in processed_data.items():
                percentage = (values['students'] / total_students * 100) if total_students > 0 else 0
                data.append({
                    'yearLevel': label,
                    'students': values['students'],
                    'percentage': round(percentage, 2)
                })

            data.sort(key=lambda x: int(re.search(r'\d+', x['yearLevel']).group(0)) if re.search(r'\d+', x['yearLevel']) else 0)

        # 2️⃣ Students per Program
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

        # 3️⃣ Student Clusters (Academic Performance vs Selected Feature)
        elif graph_type == 'student_clusters':
            secondary_feature = request.query_params.get('secondary_feature', 'financial_status')
            allowed_features = [
                'financial_status', 'workload_rating', 'help_seeking',
                'personality', 'hobby_count', 'birth_order',
                'has_external_responsibilities', 'average'
            ]
            if secondary_feature not in allowed_features:
                return Response(
                    {"error": f"Invalid secondary_feature: {secondary_feature}. Allowed: {', '.join(allowed_features)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            all_processed_students = ProcessedStudent.objects.select_related('student').all()
            clusters_data = {}

            for student_record in all_processed_students:
                cluster_id = student_record.cluster
                clusters_data.setdefault(cluster_id, [])

                academic_performance = student_record.average or 0
                secondary_value = getattr(student_record, secondary_feature, 0) or 0

                clusters_data[cluster_id].append({
                    'x': round(academic_performance, 2),
                    'y': round(secondary_value, 2)
                })

            cluster_colors = {0: '#10B981', 1: '#EF4444', 2: '#2563EB', 3: '#F59E0B'}
            datasets = [
                {
                    'label': f'Cluster {cluster_id}',
                    'data': points,
                    'backgroundColor': cluster_colors.get(cluster_id, '#6B7280'),
                    'pointRadius': 4,
                }
                for cluster_id, points in clusters_data.items()
            ]
            data = {'datasets': datasets}

        # 4️⃣ PCA Scatter
        elif graph_type == 'pca_scatter':
            try:
                pc_x = int(request.query_params.get('pc_x', 1))
                pc_y = int(request.query_params.get('pc_y', 2))
            except ValueError:
                return Response({"error": "pc_x and pc_y must be integers."}, status=status.HTTP_400_BAD_REQUEST)

            if not (1 <= pc_x <= 11 and 1 <= pc_y <= 11):
                return Response({"error": "pc_x and pc_y must be between 1 and 11."}, status=status.HTTP_400_BAD_REQUEST)

            all_students = ProcessedStudent.objects.select_related('student').all()
            clusters_data = {}

            for student in all_students:
                cluster_id = student.cluster
                clusters_data.setdefault(cluster_id, [])

                x_val = getattr(student, f'pc{pc_x}_score', None)
                y_val = getattr(student, f'pc{pc_y}_score', None)

                if x_val is not None and y_val is not None:
                    clusters_data[cluster_id].append({
                        'x': round(x_val, 4),
                        'y': round(y_val, 4)
                    })

            cluster_colors = {0: '#10B981', 1: '#EF4444', 2: '#2563EB', 3: '#F59E0B'}
            datasets = [
                {
                    'label': f'Cluster {cluster_id}',
                    'data': points,
                    'backgroundColor': cluster_colors.get(cluster_id, '#6B7280'),
                    'pointRadius': 4,
                }
                for cluster_id, points in clusters_data.items()
            ]
            data = {'datasets': datasets}

        else:
            return Response({"error": "Invalid 'graph_type' provided."}, status=status.HTTP_400_BAD_REQUEST)

        return Response(data, status=status.HTTP_200_OK)
