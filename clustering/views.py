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

# NEW imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .models import CSVUpload, ProcessedStudent  # already present at top in your file
from .services.cluster_engine import FEATURES as FEATURE_DISPLAY_NAMES  # pretty labels
import numpy as np

from .services.cluster_engine import train_and_save_model

class ClusterPCATopFeaturesView(APIView):
    """
    GET /api/clustering/clusters/<int:cluster_id>/pca-top-features/?top_pcs=3&top_features=3

    cluster_id = PK (1..4). We map it to Cluster.cluster_id (0..3) internally.
    """
    permission_classes = [AllowAny]

    def get(self, request, cluster_id: int):
        top_pcs = int(request.query_params.get('top_pcs', 3))
        top_features = int(request.query_params.get('top_features', 3))
        upload_id = request.query_params.get('upload_id')

        # resolve latest completed upload if none provided
        if upload_id:
            csv_upload = get_object_or_404(CSVUpload, id=upload_id)
        else:
            csv_upload = CSVUpload.objects.filter(processing_status='completed')\
                .order_by('-processed_timestamp', '-upload_timestamp').first()
            if not csv_upload:
                return Response({"error": "No completed CSV uploads found."}, status=400)

        # translate PK → zero-based label
        cluster_obj = get_object_or_404(Cluster, id=cluster_id)
        label = cluster_obj.cluster_id  # 0..3

        field_order = [
            'academic_performance_change','workload_rating',
            'learning_visual','learning_auditory','learning_reading_writing','learning_kinesthetic',
            'help_seeking','personality','hobby_count','financial_status',
            'birth_order','has_external_responsibilities','average',
            'marital_separated','marital_together',
        ]

        qs = ProcessedStudent.objects.filter(csv_upload=csv_upload, cluster=label).only(*field_order, 'cluster')
        rows = [[getattr(ps, f) or 0 for f in field_order] for ps in qs]
        X = np.array(rows, dtype=float)

        if X.shape[0] < 2 or X.shape[1] < 2:
            return Response({
                "upload_id": str(csv_upload.id),
                "cluster_id": cluster_id,
                "pcs": [],
                "note": "Not enough data to compute PCA for this cluster."
            }, status=200)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_components = min(top_pcs, Xs.shape[1])
        pca = PCA(n_components=n_components).fit(Xs)

        pcs_payload = []
        for i, load in enumerate(pca.components_):
            top_idx = np.argsort(np.abs(load))[::-1][:top_features]
            pcs_payload.append({
                "pc_number": i + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                "top_features": [
                    {
                        "feature_key": field_order[j],
                        "loading": float(load[j]),
                        "abs_loading": float(abs(load[j])),
                    }
                    for j in top_idx
                ]
            })

        return Response({
            "upload_id": str(csv_upload.id),
            "cluster_id": cluster_id,  # return PK, not label
            "pcs": pcs_payload
        }, status=200)

class TopPCAFeaturesPerClusterView(APIView):
    """
    GET /clustering/top-pca-features/?upload_id=<uuid>&top_pcs=3&top_features=3

    For each cluster in this upload, fit a PCA on that cluster's feature matrix,
    then return the top N features (by absolute loading) for the first K PCs.
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        upload_id = request.query_params.get('upload_id')
        if not upload_id:
            return Response({"error": "Missing required query param 'upload_id'."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Parse optional params
        try:
            top_pcs = int(request.query_params.get('top_pcs', 3))
            top_features = int(request.query_params.get('top_features', 3))
        except ValueError:
            return Response({"error": "top_pcs and top_features must be integers."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Ensure the upload exists and is completed
        csv_upload = get_object_or_404(CSVUpload, id=upload_id)
        if csv_upload.processing_status != 'completed':
            return Response({"error": f"CSV processing not completed (status={csv_upload.processing_status})."},
                            status=status.HTTP_400_BAD_REQUEST)

        # IMPORTANT: Field order must match how you build PCA inputs elsewhere
        # This mirrors CSVProcessor's feature order for PCA. :contentReference[oaicite:3]{index=3}
        field_order = [
            'academic_performance_change',
            'workload_rating',
            'learning_visual',
            'learning_auditory',
            'learning_reading_writing',
            'learning_kinesthetic',
            'help_seeking',
            'personality',
            'hobby_count',
            'financial_status',
            'birth_order',
            'has_external_responsibilities',
            'average',
            'marital_separated',
            'marital_together',
        ]

        # Human-readable names aligned to the same order.
        # These come from your clustering engine's FEATURES (same dimensionality; labels are friendly). :contentReference[oaicite:4]{index=4}
        # We trim/align them to exactly the above 15 fields:
        display_names = FEATURE_DISPLAY_NAMES[:]  # 15 names

        # Gather data for this upload
        ps_qs = ProcessedStudent.objects.filter(csv_upload=csv_upload).only(
            *field_order, 'cluster'
        )

        # Bucket by cluster id (int stored on ProcessedStudent). :contentReference[oaicite:5]{index=5}
        clusters = {}
        for ps in ps_qs:
            clusters.setdefault(ps.cluster, []).append([
                getattr(ps, f, None) or 0 for f in field_order
            ])

        results = {
            "upload_id": str(csv_upload.id),
            "top_pcs": top_pcs,
            "top_features_per_pc": top_features,
            "clusters": {}
        }

        for cluster_id, rows in clusters.items():
            X = np.array(rows, dtype=float)
            # Need at least 2 samples and 2 features to fit PCA meaningfully
            if X.shape[0] < 2 or X.shape[1] < 2:
                results["clusters"][cluster_id] = {
                    "note": "Not enough data to compute PCA for this cluster.",
                    "pcs": []
                }
                continue

            # Scale then PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_components = min(top_pcs, X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)

            # components_: shape (n_components, n_features)
            comps = pca.components_
            evr = pca.explained_variance_ratio_.tolist()

            pcs_payload = []
            for pc_idx in range(comps.shape[0]):
                loadings = comps[pc_idx]
                # top indices by absolute loading
                top_idx = np.argsort(np.abs(loadings))[::-1][:top_features]
                top_list = []
                for idx in top_idx:
                    top_list.append({
                        "feature_key": field_order[idx],
                        "feature_name": display_names[idx] if idx < len(display_names) else field_order[idx],
                        "loading": float(loadings[idx]),
                        "abs_loading": float(abs(loadings[idx])),
                    })

                pcs_payload.append({
                    "pc_number": pc_idx + 1,
                    "explained_variance_ratio": evr[pc_idx] if pc_idx < len(evr) else None,
                    "top_features": top_list,
                })

            results["clusters"][cluster_id] = {
                "samples": int(X.shape[0]),
                "pcs": pcs_payload
            }

        return Response(results, status=status.HTTP_200_OK)

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
    """
    POST /api/clustering/cluster-students/
    JSON body:
    {
      "students": [
        { "<feature>": <value>, ... }, ...
      ]
    }

    Behavior:
      - ✅ Retrains model on provided list
      - Returns clustered batch
    """
    permission_classes = [AllowAny]

    def post(self, request):
        students = request.data.get("students", [])
        if not isinstance(students, list) or not students:
            return Response({"error": "Provide non-empty 'students' list."},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            # Retrain on this set, then return the clustered batch
            _, _, _, clustered, pca_var = train_and_save_model(students, n_clusters=4, random_state=42)
            return Response(
                {
                    "success": True,
                    "total_students": len(clustered),
                    "clustered": clustered,
                    "pca_explained_variance": pca_var,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CSVUploadView(APIView):
    """
    POST /api/clustering/upload-csv/
    Form-data: csv_file: <file>
    Behavior:
      - Saves CSVUpload row
      - Runs preprocessing
      - ✅ Retrains model on this upload
      - Returns clustering results + summary
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = CSVUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        csv_upload = serializer.save(
            original_filename=request.FILES['csv_file'].name if 'csv_file' in request.FILES else ''
        )


        processor = CSVProcessor(csv_upload)
        result = processor.process_and_cluster()

        http_status = status.HTTP_200_OK if result.get("success") else status.HTTP_500_INTERNAL_SERVER_ERROR
        return Response(result, status=http_status)

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
