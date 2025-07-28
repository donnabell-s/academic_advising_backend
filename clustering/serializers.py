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
    cluster = serializers.PrimaryKeyRelatedField(
        queryset=Cluster.objects.filter(advisor__isnull=True),
        allow_null=True,
        required=False,
    )

    class Meta:
        model = Advisor
        fields = ['id', 'advisor_id', 'name', 'email', 'cluster']

    def create(self, validated_data):
        # pop off the cluster PK so we can attach it manually
        cluster_obj = validated_data.pop('cluster', None)
        advisor = super().create(validated_data)
        if cluster_obj is not None:
            cluster_obj.advisor = advisor
            cluster_obj.save()
        return advisor

    def update(self, instance, validated_data):
        # 1) pull out the new cluster PK (might be None)
        new_cluster = validated_data.pop('cluster', None)

        # 2) remember the old cluster (reverse OneToOne)
        old_cluster = getattr(instance, 'cluster', None)

        # 3) update the Advisor's basic fields
        advisor = super().update(instance, validated_data)

        # 4) if they moved off the old cluster, clear it
        if old_cluster and (new_cluster is None or new_cluster.pk != old_cluster.pk):
            old_cluster.advisor = None
            old_cluster.save()

        # 5) if they selected a new cluster, assign it
        if new_cluster is not None:
            new_cluster.advisor = advisor
            new_cluster.save()

        return advisor

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
