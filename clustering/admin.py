from django.contrib import admin
from .models import CSVUpload, ProcessedStudent

@admin.register(CSVUpload)
class CSVUploadAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'uploaded_by', 'processing_status', 
                   'total_students_processed', 'upload_timestamp', 'processed_timestamp']
    list_filter = ['processing_status', 'upload_timestamp', 'processed_timestamp']
    search_fields = ['original_filename', 'uploaded_by__username']
    readonly_fields = ['id', 'upload_timestamp', 'processed_timestamp']
    
    fieldsets = (
        ('File Information', {
            'fields': ('id', 'original_filename', 'csv_file', 'uploaded_by')
        }),
        ('Processing Status', {
            'fields': ('processing_status', 'total_students_processed', 'processing_error')
        }),
        ('Timestamps', {
            'fields': ('upload_timestamp', 'processed_timestamp')
        }),
    )

@admin.register(ProcessedStudent)
class ProcessedStudentAdmin(admin.ModelAdmin):
    list_display = ['csv_upload', 'cluster', 'academic_performance_change', 
                   'workload_rating', 'financial_status', 'processed_timestamp']
    list_filter = ['cluster', 'csv_upload', 'processed_timestamp']
    search_fields = ['csv_upload__original_filename']
    readonly_fields = ['processed_timestamp']
    
    fieldsets = (
        ('Upload Information', {
            'fields': ('csv_upload', 'cluster', 'processed_timestamp')
        }),
        ('Academic Data', {
            'fields': ('academic_performance_change', 'workload_rating', 'average')
        }),
        ('Learning Preferences', {
            'fields': ('learning_visual', 'learning_auditory', 'learning_reading_writing', 'learning_kinesthetic')
        }),
        ('Personal Information', {
            'fields': ('help_seeking', 'personality', 'hobby_count', 'financial_status', 
                      'birth_order', 'has_external_responsibilities')
        }),
        ('Family Status', {
            'fields': ('marital_separated', 'marital_together')
        }),
    )
