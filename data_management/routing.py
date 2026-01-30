from django.urls import re_path
from .consumers import TrainingProgressConsumer,EvaluationProgressConsumer

websocket_urlpatterns = [
    re_path(r'ws/training_progress/(?P<project_id>\d+)/$', TrainingProgressConsumer.as_asgi()),
    re_path(r'ws/evaluation_progress/(?P<project_id>\d+)/$', EvaluationProgressConsumer.as_asgi()),

]