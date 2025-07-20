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


class ProcessedStudent(models.Model):
    """Model to store processed student data and clustering results"""
    csv_upload = models.ForeignKey(CSVUpload, on_delete=models.CASCADE, related_name='processed_students')
    
    # Original student data (key features used for clustering)
    # Using FloatField for all numeric fields to handle NaN values properly
    academic_performance_change = models.FloatField(null=True, blank=True)
    workload_rating = models.FloatField(null=True, blank=True)
    learning_visual = models.FloatField(null=True, blank=True, default=0)
    learning_auditory = models.FloatField(null=True, blank=True, default=0)
    learning_reading_writing = models.FloatField(null=True, blank=True, default=0)
    learning_kinesthetic = models.FloatField(null=True, blank=True, default=0)
    help_seeking = models.FloatField(null=True, blank=True)
    personality = models.FloatField(null=True, blank=True)
    hobby_count = models.FloatField(null=True, blank=True)
    financial_status = models.FloatField(null=True, blank=True)
    birth_order = models.FloatField(null=True, blank=True)
    has_external_responsibilities = models.FloatField(null=True, blank=True, default=0)
    average = models.FloatField(null=True, blank=True)
    marital_separated = models.FloatField(null=True, blank=True, default=0)
    marital_together = models.FloatField(null=True, blank=True, default=0)
    
    # Clustering result
    cluster = models.IntegerField(null=True, blank=True)
    
    # Metadata
    processed_timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['cluster', 'processed_timestamp']
    
    def __str__(self):
        return f"Student in Cluster {self.cluster} from {self.csv_upload.original_filename}"
