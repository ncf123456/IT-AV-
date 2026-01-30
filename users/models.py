# users/models.py
import random
import string
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth import get_user_model


class CustomUser(AbstractUser):
    nickname = models.CharField(max_length=100, blank=True)
    gender = models.CharField(max_length=10, choices=[('M', 'Male'), ('F', 'Female')], blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    coding_experience = models.PositiveIntegerField(null=True, blank=True)
    bio = models.TextField(blank=True)

    # 指定不同的 related_name
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name='customuser_set',  # 修改这里
        related_query_name='user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='customuser_set',  # 修改这里
        related_query_name='user',
    )

    def __str__(self):
        return self.username
    


User = get_user_model()

def generate_project_number():
    while True:
        project_number = ''.join(random.choices(string.digits, k=6))
        if not Project.objects.filter(project_number=project_number).exists():
            return project_number

class Project(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    project_number = models.CharField(max_length=6, unique=True, default=generate_project_number)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name