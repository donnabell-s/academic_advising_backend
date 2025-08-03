import os
import time
import pandas as pd
import numpy as np
from django.utils import timezone
from django.conf import settings
from ..models import CSVUpload, ProcessedStudent, Student, Cluster
from ..task import preprocess_pipeline
from .cluster_engine import cluster_students
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class CSVProcessor:
    """Service class to handle CSV processing and clustering"""

    def __init__(self, csv_upload_instance):
        self.csv_upload = csv_upload_instance

    def process_and_cluster(self):
        """
        Complete pipeline: preprocess CSV data and apply clustering
        Returns: Dictionary with processing results
        """
        start_time = time.time()

        try:
            # Update status to processing
            self.csv_upload.processing_status = 'processing'
            self.csv_upload.save()

            csv_file_path = self.csv_upload.csv_file.path

            # Step 0: Ensure 4 default clusters exist
            if Cluster.objects.count() < 4:
                default_clusters_data = [
                    {'cluster_id': 0, 'name': 'Cluster Alpha', 'description': 'Students with high academic performance and low workload.'},
                    {'cluster_id': 1, 'name': 'Cluster Beta', 'description': 'Students focused on visual learning with external responsibilities.'},
                    {'cluster_id': 2, 'name': 'Cluster Gamma', 'description': 'Students with high help-seeking behavior and diverse hobbies.'},
                    {'cluster_id': 3, 'name': 'Cluster Delta', 'description': 'Students with balanced characteristics.'},
                ]
                for data in default_clusters_data:
                    Cluster.objects.get_or_create(cluster_id=data['cluster_id'], defaults=data)

            # Step 1: Preprocess CSV
            students_list_preprocessed, label_encoders, scaler = preprocess_pipeline(
                csv_file_path,
                return_format='api'
            )

            # Step 1.5: Clean NaN values
            cleaned_students = []
            for i, student in enumerate(students_list_preprocessed):
                for key, value in student.items():
                    if pd.isna(value) or (isinstance(value, float) and value != value):
                        if 'Learning_' in key or 'Marital_' in key or 'Has External' in key:
                            student[key] = 0
                        elif key in ['Academic Performance Change', 'Workload Rating', 'Help Seeking', 'Personality', 'Financial Status', 'Average']:
                            student[key] = 0.0
                        else:
                            student[key] = None
                cleaned_students.append(student)
            students_list_preprocessed = cleaned_students

            # Step 2: Apply clustering
            clustered_students_data = cluster_students(students_list_preprocessed)

            # Step 3: Create Student & ProcessedStudent records
            processed_students_to_create = []

            def safe_convert(value, default=None):
                if pd.isna(value) or value != value:
                    return default
                return float(value) if isinstance(value, (int, float)) else value

            for student_data in clustered_students_data:
                student_id = student_data.get('Student ID')
                student_name = student_data.get('Name')
                raw_program_grade = student_data.get('Program and Year')
                program_grade = str(raw_program_grade or "N/A0").strip()[:50]
                assigned_cluster_int = safe_convert(student_data.get('cluster'), -1)

                cluster_obj = Cluster.objects.filter(cluster_id=int(assigned_cluster_int)).first() if assigned_cluster_int != -1 else None

                student_obj = None
                if student_id:
                    student_obj, _ = Student.objects.update_or_create(
                        student_id=student_id,
                        defaults={
                            'name': student_name,
                            'program_and_grade': program_grade,
                            'cluster': cluster_obj
                        }
                    )

                processed_students_to_create.append(
                    ProcessedStudent(
                        csv_upload=self.csv_upload,
                        student=student_obj,
                        academic_performance_change=safe_convert(student_data.get('Academic Performance Change')),
                        workload_rating=safe_convert(student_data.get('Workload Rating')),
                        learning_visual=safe_convert(student_data.get('Learning_Visual'), 0),
                        learning_auditory=safe_convert(student_data.get('Learning_Auditory'), 0),
                        learning_reading_writing=safe_convert(student_data.get('Learning_Reading/Writing'), 0),
                        learning_kinesthetic=safe_convert(student_data.get('Learning_Kinesthetic'), 0),
                        help_seeking=safe_convert(student_data.get('Help Seeking')),
                        personality=safe_convert(student_data.get('Personality')),
                        hobby_count=safe_convert(student_data.get('Hobby Count')),
                        financial_status=safe_convert(student_data.get('Financial Status')),
                        birth_order=safe_convert(student_data.get('Birth Order')),
                        has_external_responsibilities=safe_convert(student_data.get('Has External Responsibilities'), 0),
                        average=safe_convert(student_data.get('Average')),
                        marital_separated=safe_convert(student_data.get('Marital_Separated'), 0),
                        marital_together=safe_convert(student_data.get('Marital_Together'), 0),
                        cluster=assigned_cluster_int
                    )
                )

            # Bulk create processed students
            ProcessedStudent.objects.bulk_create(processed_students_to_create)

            # ✅ Step 4: Compute PCA scores
            processed_students = list(ProcessedStudent.objects.filter(csv_upload=self.csv_upload).order_by('id'))

            feature_matrix = []
            for ps in processed_students:
                feature_matrix.append([
                    ps.academic_performance_change or 0,
                    ps.workload_rating or 0,
                    ps.learning_visual or 0,
                    ps.learning_auditory or 0,
                    ps.learning_reading_writing or 0,
                    ps.learning_kinesthetic or 0,
                    ps.help_seeking or 0,
                    ps.personality or 0,
                    ps.hobby_count or 0,
                    ps.financial_status or 0,
                    ps.birth_order or 0,
                    ps.has_external_responsibilities or 0,
                    ps.average or 0,
                    ps.marital_separated or 0,
                    ps.marital_together or 0,
                ])

            X = np.array(feature_matrix)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if X.shape[0] > 0 and X.shape[1] >= 2:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_components = min(11, X_scaled.shape[1])
                pca = PCA(n_components=n_components)
                pca_scores = pca.fit_transform(X_scaled)

                for ps, scores in zip(processed_students, pca_scores):
                    for i in range(n_components):
                        setattr(ps, f'pc{i+1}_score', float(scores[i]))
                    ps.save(update_fields=[f'pc{i+1}_score' for i in range(n_components)])

            # Step 5: Mark processing complete
            self.csv_upload.processing_status = 'completed'
            self.csv_upload.total_students_processed = len(clustered_students_data)
            self.csv_upload.processed_timestamp = timezone.now()
            self.csv_upload.save()

            processing_time = time.time() - start_time

            return {
                'success': True,
                'upload_id': self.csv_upload.id,
                'total_students': len(clustered_students_data),
                'clusters_summary': self._generate_cluster_summary(clustered_students_data),
                'processing_time': processing_time,
                'students_by_cluster': self._group_students_by_cluster(clustered_students_data),
                'clustered_data': clustered_students_data,
                'pca_explained_variance': pca.explained_variance_ratio_.tolist() if X.shape[0] > 0 and X.shape[1] >= 2 else []
            }

        except Exception as e:
            self.csv_upload.processing_status = 'failed'
            self.csv_upload.processing_error = str(e)
            self.csv_upload.processed_timestamp = timezone.now()
            self.csv_upload.save()
            return {
                'success': False,
                'error': str(e),
                'upload_id': self.csv_upload.id
            }

    def _generate_cluster_summary(self, clustered_students):
        clusters = {}
        for student in clustered_students:
            cluster_id = student['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'count': 0,
                    'avg_performance_change': 0,
                    'avg_workload_rating': 0,
                    'avg_financial_status': 0,
                    'avg_hobby_count': 0
                }
            clusters[cluster_id]['count'] += 1

        for student in clustered_students:
            cluster_id = student['cluster']
            count = clusters[cluster_id]['count']
            if count > 0:
                clusters[cluster_id]['avg_performance_change'] += student.get('Academic Performance Change', 0) / count
                clusters[cluster_id]['avg_workload_rating'] += student.get('Workload Rating', 0) / count
                clusters[cluster_id]['avg_financial_status'] += student.get('Financial Status', 0) / count
                clusters[cluster_id]['avg_hobby_count'] += student.get('Hobby Count', 0) / count

        return clusters

    def _group_students_by_cluster(self, clustered_students):
        students_by_cluster = {}
        for student in clustered_students:
            cluster_id = student['cluster']
            if cluster_id not in students_by_cluster:
                students_by_cluster[cluster_id] = []
            students_by_cluster[cluster_id].append(student)
        return students_by_cluster

    @staticmethod
    def get_processing_status(upload_id):
        try:
            csv_upload = CSVUpload.objects.get(id=upload_id)
            return {
                'status': csv_upload.processing_status,
                'total_students': csv_upload.total_students_processed,
                'error': csv_upload.processing_error,
                'upload_timestamp': csv_upload.upload_timestamp,
                'processed_timestamp': csv_upload.processed_timestamp
            }
        except CSVUpload.DoesNotExist:
            return {'status': 'not_found'}

    @staticmethod
    def get_clustering_results(upload_id):
        try:
            csv_upload = CSVUpload.objects.get(id=upload_id)
            if csv_upload.processing_status != 'completed':
                return {'error': 'Processing not completed'}

            processed_students = ProcessedStudent.objects.filter(csv_upload=csv_upload).select_related('student')

            clusters = {}
            for student in processed_students:
                clusters.setdefault(student.cluster, []).append({
                    'student_id': student.student.student_id if student.student else 'N/A',
                    'student_name': student.student.name if student.student else 'N/A',
                    'program_and_grade': student.student.program_and_grade if student.student else 'N/A',
                    'academic_performance_change': student.academic_performance_change,
                    'workload_rating': student.workload_rating,
                    'learning_visual': student.learning_visual,
                    'learning_auditory': student.learning_auditory,
                    'learning_reading_writing': student.learning_reading_writing,
                    'learning_kinesthetic': student.learning_kinesthetic,
                    'help_seeking': student.help_seeking,
                    'personality': student.personality,
                    'hobby_count': student.hobby_count,
                    'financial_status': student.financial_status,
                    'birth_order': student.birth_order,
                    'has_external_responsibilities': student.has_external_responsibilities,
                    'average': student.average,
                    'marital_separated': student.marital_separated,
                    'marital_together': student.marital_together,
                    'cluster': student.cluster
                })

            return {
                'upload_id': upload_id,
                'total_students': len(processed_students),
                'clusters': clusters,
                'upload_info': {
                    'filename': csv_upload.original_filename,
                    'upload_timestamp': csv_upload.upload_timestamp,
                    'processed_timestamp': csv_upload.processed_timestamp
                }
            }

        except CSVUpload.DoesNotExist:
            return {'error': 'Upload not found'}
