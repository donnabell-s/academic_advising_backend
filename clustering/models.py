from django.db import models
from django.contrib.auth.models import User
import uuid

class CSVUpload(models.Model):
    """Model to track CSV file uploads and processing status"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='csv_uploads')
    original_filename = models.CharField(max_length=255)
    csv_file = models.FileField(upload_to='uploaded_csvs/')
    upload_timestamp = models.DateTimeField(auto_now_add=True)

    # Processing status
    PROCESSING_STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    processing_status = models.CharField(
        max_length=20,
        choices=PROCESSING_STATUS_CHOICES,
        default='uploaded'
    )

    # Results
    total_students_processed = models.IntegerField(null=True, blank=True)
    processing_error = models.TextField(null=True, blank=True)
    processed_timestamp = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-upload_timestamp']

    def __str__(self):
        return f"{self.original_filename} - {self.processing_status}"

# Cluster Model - Defined before Student as Student will link to it
class Cluster(models.Model):
    # Changed to IntegerField
    cluster_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    advisor = models.OneToOneField(
        'Advisor',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='cluster'
    )

    def __str__(self):
        return self.name

    @property
    def student_count(self):
        # This will count ProcessedStudent objects linked to a cluster
        return ProcessedStudent.objects.filter(cluster=self.cluster_id).count()

# Student Model
class Student(models.Model):
    student_id = models.CharField(max_length=15, unique=True)
    name = models.CharField(max_length=100, null=True, blank=True)
    program_and_grade = models.CharField(max_length=50)
    cluster = models.ForeignKey(
        Cluster, # Refer directly to the Cluster model
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='students'
    )

    def __str__(self):
        return f"{self.student_id} - {self.name or 'Unnamed'}"

# ProcessedStudent Model
class ProcessedStudent(models.Model):
    """Model to store individual student data after CSV processing and clustering"""
    csv_upload = models.ForeignKey(CSVUpload, on_delete=models.CASCADE, related_name='processed_students')
    # Link to the Student object
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='processed_records', null=True, blank=True)

    # Key features extracted/derived from CSV
    academic_performance_change = models.FloatField(null=True, blank=True)
    workload_rating = models.FloatField(null=True, blank=True)
    learning_visual = models.IntegerField(null=True, blank=True, default=0)
    learning_auditory = models.IntegerField(null=True, blank=True, default=0)
    learning_reading_writing = models.IntegerField(null=True, blank=True, default=0)
    learning_kinesthetic = models.IntegerField(null=True, blank=True, default=0)
    help_seeking = models.FloatField(null=True, blank=True)
    personality = models.FloatField(null=True, blank=True) # This should ideally be an IntegerField if encoded
    hobby_count = models.IntegerField(null=True, blank=True)
    financial_status = models.FloatField(null=True, blank=True)
    birth_order = models.IntegerField(null=True, blank=True) # This should ideally be an IntegerField if encoded
    has_external_responsibilities = models.IntegerField(null=True, blank=True, default=0)
    average = models.FloatField(null=True, blank=True, default=0)
    marital_separated = models.IntegerField(null=True, blank=True, default=0)
    marital_together = models.IntegerField(null=True, blank=True, default=0)

    # Clustering result (this remains an IntegerField as it's the raw cluster output)
    cluster = models.IntegerField(null=True, blank=True)

    # Metadata
    processed_timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['cluster', 'processed_timestamp']

    def __str__(self):
        return f"Student {self.student.student_id if self.student else 'N/A'} in Cluster {self.cluster} from {self.csv_upload.original_filename}"

# Advisor Model
class Advisor(models.Model):
    advisor_id = models.CharField(max_length=15, unique=True, null=True, blank=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.name
