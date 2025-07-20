from django.db import models

# Advisor Model
class Advisor(models.Model):
    advisor_id = models.CharField(max_length=15, unique=True, null=True, blank=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    # An advisor can be unassigned (null cluster)
    # One-to-one from Cluster will handle "cluster under" relationship

    def __str__(self):
        return self.name

# Cluster Model
class Cluster(models.Model):
    cluster_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    advisor = models.OneToOneField(
        Advisor,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='cluster'
    )

    def __str__(self):
        return self.name

    @property
    def student_count(self):
        return self.students.count()  

# Student Model
class Student(models.Model):
    student_id = models.CharField(max_length=15, unique=True)
    name = models.CharField(max_length=100, null=True, blank=True) 
    program_and_grade = models.CharField(max_length=50)  
    cluster = models.ForeignKey(
        'Cluster',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='students'
    )

    def __str__(self):
        return f"{self.student_id} - {self.name or 'Unnamed'}"
