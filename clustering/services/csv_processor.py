import os
import time
import pandas as pd
from django.utils import timezone
from django.conf import settings
from ..models import CSVUpload, ProcessedStudent
from ..task import preprocess_pipeline
from .cluster_engine import cluster_students
import tempfile

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
            
            # Get the file path
            csv_file_path = self.csv_upload.csv_file.path
            
            # Step 1: Preprocess the data using the task pipeline
            students_list, label_encoders, scaler = preprocess_pipeline(
                csv_file_path, 
                return_format='api'
            )
            
            # Step 1.5: Clean and validate the preprocessed data
            print(f"DEBUG: Received {len(students_list)} students from preprocessing")
            cleaned_students = []
            for i, student in enumerate(students_list):
                # Check for NaN values and log them
                nan_fields = []
                for key, value in student.items():
                    if pd.isna(value) or (isinstance(value, float) and value != value):
                        nan_fields.append(key)
                
                if nan_fields:
                    print(f"DEBUG: Student {i} has NaN values in: {nan_fields}")
                    # Replace NaN with appropriate defaults
                    for field in nan_fields:
                        if 'Learning_' in field or 'Marital_' in field or 'Has External' in field:
                            student[field] = 0
                        else:
                            student[field] = 0.0
                
                cleaned_students.append(student)
            
            print(f"DEBUG: Cleaned {len(cleaned_students)} students")
            students_list = cleaned_students
            
            # Step 2: Apply clustering
            clustered_students = cluster_students(students_list)
            
            # Step 3: Save processed students to database
            processed_students = []
            for student_data in clustered_students:
                # Helper function to convert NaN to None for database storage
                def safe_convert(value, default=None):
                    if pd.isna(value) or value != value:  # Check for NaN
                        return default
                    return float(value) if isinstance(value, (int, float)) else value
                
                processed_student = ProcessedStudent(
                    csv_upload=self.csv_upload,
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
                    cluster=safe_convert(student_data.get('cluster'), -1)
                )
                processed_students.append(processed_student)
            
            # Bulk create for efficiency
            ProcessedStudent.objects.bulk_create(processed_students)
            
            # Step 4: Generate summary statistics
            clusters_summary = self._generate_cluster_summary(clustered_students)
            students_by_cluster = self._group_students_by_cluster(clustered_students)
            
            # Update CSV upload record
            self.csv_upload.processing_status = 'completed'
            self.csv_upload.total_students_processed = len(clustered_students)
            self.csv_upload.processed_timestamp = timezone.now()
            self.csv_upload.save()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'upload_id': self.csv_upload.id,
                'total_students': len(clustered_students),
                'clusters_summary': clusters_summary,
                'processing_time': processing_time,
                'students_by_cluster': students_by_cluster,
                'clustered_data': clustered_students
            }
            
        except Exception as e:
            # Update status to failed
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
        """Generate summary statistics for each cluster"""
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
        
        # Calculate averages
        for student in clustered_students:
            cluster_id = student['cluster']
            count = clusters[cluster_id]['count']
            clusters[cluster_id]['avg_performance_change'] += student.get('Academic Performance Change', 0) / count
            clusters[cluster_id]['avg_workload_rating'] += student.get('Workload Rating', 0) / count
            clusters[cluster_id]['avg_financial_status'] += student.get('Financial Status', 0) / count
            clusters[cluster_id]['avg_hobby_count'] += student.get('Hobby Count', 0) / count
        
        return clusters
    
    def _group_students_by_cluster(self, clustered_students):
        """Group students by their assigned cluster"""
        students_by_cluster = {}
        for student in clustered_students:
            cluster_id = student['cluster']
            if cluster_id not in students_by_cluster:
                students_by_cluster[cluster_id] = []
            students_by_cluster[cluster_id].append(student)
        
        return students_by_cluster
    
    @staticmethod
    def get_processing_status(upload_id):
        """Get the current processing status of a CSV upload"""
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
        """Get the clustering results for a processed CSV"""
        try:
            csv_upload = CSVUpload.objects.get(id=upload_id)
            if csv_upload.processing_status != 'completed':
                return {'error': 'Processing not completed'}
            
            processed_students = ProcessedStudent.objects.filter(csv_upload=csv_upload)
            
            # Group by clusters
            clusters = {}
            for student in processed_students:
                cluster_id = student.cluster
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append({
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
