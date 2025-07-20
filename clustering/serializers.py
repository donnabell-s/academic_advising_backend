from rest_framework import serializers
from .models import Advisor, Cluster, Student

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
