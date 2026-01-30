
# data_management/models.py
from django.db import models
from users.models import Project

class DataFile(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    csv_data = models.TextField()

    def __str__(self):
        return self.filename

class ModelFile(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    h5_data = models.BinaryField()

    def __str__(self):
        return self.filename