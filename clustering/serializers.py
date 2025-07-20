from rest_framework import serializers
from .models import CSVUpload, ProcessedStudent

class CSVUploadSerializer(serializers.ModelSerializer):
    """Serializer for CSV file uploads"""
    class Meta:
        model = CSVUpload
        fields = ['id', 'original_filename', 'csv_file', 'upload_timestamp', 
                 'processing_status', 'total_students_processed', 'processing_error']
        read_only_fields = ['id', 'upload_timestamp', 'processing_status', 
                           'total_students_processed', 'processing_error']

class ProcessedStudentSerializer(serializers.ModelSerializer):
    """Serializer for processed student data"""
    class Meta:
        model = ProcessedStudent
        fields = '__all__'

class ClusteringResultSerializer(serializers.Serializer):
    """Serializer for clustering results"""
    upload_id = serializers.UUIDField()
    total_students = serializers.IntegerField()
    clusters_summary = serializers.DictField()
    processing_time = serializers.FloatField()
    students_by_cluster = serializers.DictField()