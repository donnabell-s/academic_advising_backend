from rest_framework import serializers
from .models import CSVUpload, ProcessedStudent
from rest_framework import serializers
from .models import Advisor, Cluster, Student

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
class AdvisorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Advisor
        fields = '__all__'

class ClusterSerializer(serializers.ModelSerializer):
    advisor_name = serializers.CharField(source='advisor.name', read_only=True)
    student_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Cluster
        fields = ['id', 'cluster_id', 'name', 'description', 'advisor', 'advisor_name', 'student_count']

class StudentSerializer(serializers.ModelSerializer):
    cluster_name = serializers.CharField(source='cluster.name', read_only=True)

    class Meta:
        model = Student
        fields = ['id', 'student_id', 'name', 'program_and_grade', 'cluster', 'cluster_name']
