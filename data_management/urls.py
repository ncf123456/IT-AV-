# data_management/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('manage_data/<int:project_id>/', views.manage_data, name='manage_data'),
    path('upload_data/<int:project_id>/', views.upload_data, name='upload_data'),
    path('preview_data/<int:project_id>/<str:data_filename>/', views.preview_data, name='preview_data'),
    path('sample_data/<int:project_id>/<str:data_filename>/', views.sample_data, name='sample_data'),   
    path('delete_data/<int:project_id>/<str:data_filename>/', views.delete_data, name='delete_data'),
    path('model_management/<int:project_id>/', views.model_management, name='model_management'),
    path('upload_model/<int:project_id>/', views.upload_model, name='upload_model'),
    path('delete_model/<int:project_id>/<str:model_filename>/', views.delete_model, name='delete_model'),
    path('data_training/<int:project_id>/', views.data_training, name='data_training'),
    path('get_columns/<int:project_id>/<str:data_filename>/', views.get_columns, name='get_columns'),  
    path('data_prediction/<int:project_id>/', views.data_prediction, name='data_prediction'),
    path('visualization_results/<int:project_id>/', views.visualization_results, name='visualization_results'),
    path('rename_data/<int:project_id>/<str:data_filename>/', views.rename_data, name='rename_data'),
    path('model/rename/<int:project_id>/<str:model_filename>/', views.rename_model, name='rename_model'),
    path('model/delete/<int:project_id>/<str:model_filename>/', views.delete_model, name='delete_model'),
    path('download_data/<int:project_id>/<str:data_filename>/', views.download_data, name='download_data'),  
    path('download_model/<int:project_id>/<str:model_filename>/', views.download_model, name='download_model'),
    path('download_pdf/<int:project_id>/', views.download_pdf, name='download_pdf'),

    path('save_model/<int:project_id>/', views.save_model, name='save_model'),
    path('evaluation_calculation/<int:project_id>/', views.evaluation_calculation, name='evaluation_calculation'),
    path('save_evaluation_result/<int:project_id>/', views.save_evaluation_result, name='save_evaluation_result'),
    path('data_training_dash/<int:project_id>/', views.data_training_dash, name='data_training_dash'),
    path('delete_previous_file/<int:project_id>/<str:file_type>/', views.delete_previous_file, name='delete_previous_file'),
    
    # path('hyperparameter_training/<int:project_id>/', views.hyperparameter_training, name='hyperparameter_training'),
    path('hyperparameter_tuning/<int:project_id>/', views.hyperparameter_training, name='hyperparameter_tuning'),
    
    path('get_data_for_charts/<int:project_id>/<str:data_filename>/', views.get_data_for_charts, name='get_data_for_charts'),
    path('save_hyperparameter_result/<int:project_id>/', views.save_hyperparameter_result, name='save_hyperparameter_result'),
    path('stop_hyperparameter_tuning/<int:project_id>/', views.stop_hyperparameter_tuning, name='stop_hyperparameter_tuning'),
    path('generate_histograms_and_parallel_coordinates/<int:project_id>/', views.generate_histograms_and_parallel_coordinates, name='generate_histograms_and_parallel_coordinates'),
    path('get_histogram_and_parallel_coordinates_data/<int:project_id>/<str:data_filename>/', views.get_histogram_and_parallel_coordinates_data, name='get_histogram_and_parallel_coordinates_data'),
    path('get_data_size/<int:project_id>/<str:data_filename>/', views.get_data_size, name='get_data_size'),
    path('bayesian_optimization/<int:project_id>/', views.bayesian_optimization, name='bayesian_optimization'),
    path('get_suggestions/<int:project_id>/', views.get_suggestions, name='get_suggestions'),
    path('save_prediction_result/<int:project_id>/', views.save_prediction_result, name='save_prediction_result'),
]

