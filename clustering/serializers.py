from rest_framework import serializers
from .models import CSVUpload, ProcessedStudent
from rest_framework import serializers
from .models import Advisor, Cluster, Student, PCAComponent

class PCAComponentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PCAComponent
        fields = ['id', 'pc_number', 'name', 'description']

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
    success = serializers.BooleanField()
    upload_id = serializers.UUIDField()
    total_students = serializers.IntegerField()
    clusters_summary = serializers.DictField()
    processing_time = serializers.FloatField()
    students_by_cluster = serializers.DictField()
    clustered_data = serializers.JSONField()
    error = serializers.CharField(required=False)
    student_count = serializers.IntegerField(read_only=True)


class AdvisorSerializer(serializers.ModelSerializer):
    # this field WILL show up on GET
    cluster = serializers.PrimaryKeyRelatedField(
        queryset=Cluster.objects.all(),
        allow_null=True,
        required=False,
    )
    # And this gives you the human readable name
    cluster_name = serializers.CharField(
        source='cluster.name',
        read_only=True,
        default=None
    )

    class Meta:
        model  = Advisor
        fields = ['id', 'advisor_id', 'name', 'email', 'cluster','cluster_name',]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # when editing, include the advisor’s *own* cluster in the choices
        if self.instance and getattr(self.instance, 'pk', None):
            # the reverse accessor gives you the Cluster instance (or raises)
            try:
                current = self.instance.cluster  # <— the OneToOne reverse
            except Cluster.DoesNotExist:
                current = None

            qs = Cluster.objects.filter(advisor__isnull=True)
            if current:
                qs = qs | Cluster.objects.filter(pk=current.pk)
            self.fields['cluster'].queryset = qs.distinct()

    def create(self, validated_data):
        cluster_obj = validated_data.pop('cluster', None)
        advisor     = super().create(validated_data)
        if cluster_obj:
            cluster_obj.advisor = advisor
            cluster_obj.save()
        return advisor

    def update(self, instance, validated_data):
        new_cluster = validated_data.pop('cluster', None)
        # grab old via reverse
        try:
            old_cluster = instance.cluster
        except Cluster.DoesNotExist:
            old_cluster = None

        advisor = super().update(instance, validated_data)

        # clear out the old cluster if they moved off it
        if old_cluster and (new_cluster is None or new_cluster.pk != old_cluster.pk):
            old_cluster.advisor = None
            old_cluster.save()

        # assign the new one
        if new_cluster:
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

