# data_management/views.py
from io import BytesIO
import io
import numpy as np
from io import StringIO
from tkinter import Canvas
from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from django.shortcuts import render, redirect, get_object_or_404
from users.models import Project
from .models import DataFile, ModelFile
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
import shutil
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import RenameDataFileForm
from django.contrib import messages
from humanize import naturalsize
# from .training_utils import train_model 
from .training_utils import train_model as spatio_temporal_train_model,predict_model as spatio_temporal_predict_model,train_multiple_param_combinations as spatio_temporal_param_combinations, plot_loss_web, plot_metrics_web, plot_predictions_web, plot_feature_importance_web,plot_histogram_web,plot_parallel_coordinates_web, train_with_bayesian, train_with_bayesian_optimization
from .traditional_cnn_lstm_attention import train_model as traditional_train_model,predict_model as traditional_predict_model,train_multiple_param_combinations as traditional_param_combinations,train_with_bayesian_optimization as traditional_train_with_bayesian_optimization
from fpdf import FPDF
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio
from django.conf import settings
import re
import threading

def human_readable_size(size_bytes):
    """Convert bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def manage_data(request, project_id):
    print("Handling manage_data request...")
    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    # Adding debug information
    for data_file in data_files:
        try:
            df = pd.read_json(data_file.csv_data)
            size_bytes = len(data_file.csv_data.encode('utf-8'))
            print(f"File: {data_file.filename}, Size (bytes): {size_bytes}")
            data_file.size = human_readable_size(size_bytes)
        except Exception as e:
            print(f"Error reading file {data_file.filename}: {str(e)}")
            data_file.size = "N/A"

    return render(request, 'data_management/manage_data.html', {'project': project, 'data_files': data_files})

def upload_data(request, project_id):
    print("Handling upload_data request...")
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage(location=os.path.join('media', str(project_id)))  # Store files in project-specific directory
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        project = get_object_or_404(Project, id=project_id)
        data_file = DataFile(project=project, filename=filename, csv_data=pd.read_csv(file_path).to_json())
        data_file.save()

        return redirect('manage_data', project_id=project_id)

    return redirect('manage_data', project_id=project_id)

@csrf_exempt
def rename_data(request, project_id, data_filename):
    try:
        project = get_object_or_404(Project, id=project_id)
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)

        # Get the new filename
        new_filename = request.POST.get('filename')

        if not new_filename:
            return JsonResponse({'status': 'error', 'message': '新文件名不能为空'})

        # Update filename
        data_file.filename = new_filename
        data_file.save()

        print(f"Renamed file from {data_filename} to {new_filename}")

        return JsonResponse({'status': 'success', 'message': '文件名已成功更改。'})
    except Exception as e:
        print(f"Error renaming file: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)})

def preview_data(request, project_id, data_filename):
    print("Handling preview_data request...")
    project = get_object_or_404(Project, id=project_id)
    data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
    df = pd.read_json(data_file.csv_data)
    
    # 获取表头和前50行数据
    preview_table_headings = df.columns.tolist()
    preview_table_rows = df.head(50).values.tolist()
    
    return render(request, 'data_management/preview_data.html', {
        'project': project,
        'data_file': data_file,
        'preview_table_headings': preview_table_headings,
        'preview_table_rows': preview_table_rows
    })

def sample_data(request, project_id, data_filename):
    print("Handling sample_data request...")
    if request.method == 'POST':
        project = get_object_or_404(Project, id=project_id)
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
        df = pd.read_json(data_file.csv_data)

        remove_missing = request.POST.get('remove_missing') == 'on'
        sample_percent = int(request.POST.get('sample_percent'))
        new_filename = request.POST.get('new_filename')

        # Ensure the new filename ends with .csv
        if not new_filename.endswith('.csv'):
            new_filename += '.csv'

        if remove_missing:
            # Replace '--' with NaN
            df.replace('--', pd.NA, inplace=True)
            # Remove rows with any NaN values
            df.dropna(inplace=True)

        sampled_df = df.sample(frac=sample_percent / 100.0)
        sampled_data = sampled_df.to_csv(index=False, line_terminator='\n')  # Ensure no extra empty lines


        file_obj = StringIO(sampled_data)

        fs = FileSystemStorage(location=os.path.join('media', str(project.id)))  # Store files in project-specific directory
        file_path = fs.save(new_filename, file_obj)
        full_path = fs.path(file_path)

        # Update file size
        if os.path.exists(full_path):
            size_bytes = os.path.getsize(full_path)
            new_data_file = DataFile(project=project, filename=new_filename, csv_data=sampled_df.to_json())
            new_data_file.save()
            new_data_file.size = naturalsize(size_bytes)
            new_data_file.save()
            print(f"Saved sampled data to {full_path}, Size: {size_bytes} bytes")
        else:
            print(f"Sampled file not found: {full_path}")

        return redirect('manage_data', project_id=project_id)

    return redirect('manage_data', project_id=project_id)

def delete_data(request, project_id, data_filename):
    print("Handling delete_data request...")
    project = get_object_or_404(Project, id=project_id)
    data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
    file_path = os.path.join('media', str(project.id), data_file.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    data_file.delete()
    return redirect('manage_data', project_id=project_id)

def model_management(request, project_id):
    print("Handling model_management request...")
    project = get_object_or_404(Project, id=project_id)
    model_files = ModelFile.objects.filter(project=project)
    return render(request, 'data_management/model_management.html', {'project': project, 'model_files': model_files})

@csrf_exempt
def rename_model(request, project_id, model_filename):
    try:
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

        project = get_object_or_404(Project, id=project_id)
        model_file = get_object_or_404(ModelFile, project=project, filename=model_filename)

        # Get the new filename
        new_filename = request.POST.get('filename')

        if not new_filename:
            return JsonResponse({'status': 'error', 'message': '新文件名不能为空'})

        # Update filename in the database
        model_file.filename = new_filename
        model_file.save()

        return JsonResponse({'status': 'success', 'message': '文件名已成功更改。'})
    except Exception as e:
        print(f"Error renaming file: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)})

def upload_model(request, project_id):
    print("Handling upload_model request...")
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage(location=os.path.join('media', str(project_id)))  # Store files in project-specific directory
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        project = get_object_or_404(Project, id=project_id)
        with open(file_path, 'rb') as f:
            h5_data = f.read()
        model_file = ModelFile(project=project, filename=filename, h5_data=h5_data)
        model_file.save()

        return redirect('model_management', project_id=project_id)

    return redirect('model_management', project_id=project_id)

@csrf_exempt
def delete_model(request, project_id, model_filename):
    print("Handling delete_model request...")
    try:
        project = get_object_or_404(Project, id=project_id)
        model_file = get_object_or_404(ModelFile, project=project, filename=model_filename)
        file_path = os.path.join('media', str(project.id), model_file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted model file: {file_path}")
        model_file.delete()
        return redirect('model_management', project_id=project_id)  # Redirect back to model management page
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



def data_training(request, project_id):
    print("Handling data_training request...")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")

    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    if request.method == 'POST':
        try:
            print("Received POST request in data_training...")
            print(f"POST data: {request.POST}")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            # Define parameter grid based on user input or default values
            param_grid = {}
            if request.POST.get('method') == 'spatio_temporal':
                param_grid = {
                    'lstm_units': [int(request.POST.get('st_lstm_units', 50))],
                    'dense_units': [int(request.POST.get('st_dense_units', 50))],
                    'filters': [int(request.POST.get('st_filters', 64))],
                    'temporal_kernel_size': [int(request.POST.get('st_temporal_kernel_size', 3))],
                    'spatial_kernel_sizes': [list(map(int, request.POST.get('st_spatial_kernel_sizes', '[3, 3]').strip('[]').split(',')))],
                    'dynamic_attention_units': [int(request.POST.get('st_dynamic_attention_units', 50))],
                    'epochs': [int(request.POST.get('st_epochs', 10))],
                    'batch_size': [int(request.POST.get('st_batch_size', 32))]
                }
            elif request.POST.get('method') == 'traditional':
                param_grid = {
                    'filters': [int(request.POST.get('t_filters', 64))],  
                    'kernel_size': [int(request.POST.get('t_kernel_size', 3))], 
                    'lstm_units': [int(request.POST.get('t_lstm_units', 50))],  
                    'dense_units': [int(request.POST.get('t_dense_units', 50))],  
                    'dynamic_attention_units': [int(request.POST.get('t_dynamic_attention_units', 50))],  
                    'epochs': [int(request.POST.get('t_epochs', 10))],
                    'batch_size': [int(request.POST.get('t_batch_size', 32))],  
                }

            print(f"Parsed param_grid: {param_grid}")

            # Get test size ratio from form data
            test_size_ratio = float(request.POST.get('test_size_ratio', 0.2))

            # Train the model using the provided method
            model_filename = f'model_{project_id}'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if request.POST.get('method') == 'spatio_temporal':
                future = loop.run_in_executor(None, spatio_temporal_train_model, df, target_columns, standard_date_features, time_features, geographic_feature, other_features, param_grid, project_id, model_filename, test_size_ratio)
            elif request.POST.get('method') == 'traditional':
                future = loop.run_in_executor(None, traditional_train_model, df, target_columns, standard_date_features, other_features, param_grid, project_id, model_filename, test_size_ratio)
            else:
                return JsonResponse({'status': 'error', 'message': '无效的训练方法'}, status=400)

            result = loop.run_until_complete(future)
            loop.close()

            if isinstance(result, dict):
                for key in ['rmse', 'mae', 'r2', 'adj_r2', 'nse']:
                    if key in result and isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()

                # Convert numpy.float32 to float
                for key, value in result.items():
                    if isinstance(value, np.float32):
                        result[key] = float(value)

            if result['status'] == 'success':
                print("Model saved successfully.")
                
                # 返回早停触发的轮次信息
                response_data = {
                    'status': 'success',
                    'message': '模型训练并保存成功！',
                    'triggered_epoch': result.get('triggered_epoch', None)
                }
                return JsonResponse(response_data, status=200)
            
            else:
                response_data = {
                    'status': 'error',
                    'message': '模型训练失败'
                }
                return JsonResponse(response_data, status=500)

        except Exception as e:
            print(f"Error during data_training: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Prepare form options for rendering
    if data_files.exists():
        columns = list(pd.read_json(data_files.first().csv_data).columns)
    else:
        columns = []

    return render(request, 'data_management/data_training.html', {
        'project': project,
        'data_files': data_files,
        'columns': columns
    })




@csrf_exempt
def evaluation_calculation(request, project_id):
    print("Handling evaluation_calculation request...")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")

    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    if request.method == 'POST':
        try:
            print("Received POST request in evaluation_calculation...")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and not time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            # Define parameter grid based on user input or default values
            param_grid = {}
            if request.POST.get('method') == 'spatio_temporal':
                st_spatial_kernel_sizes_str = request.POST.get('st_spatial_kernel_sizes', '[[3, 3], [4, 4]]')
                st_spatial_kernel_sizes = eval(st_spatial_kernel_sizes_str) 

                print(f"Received st_spatial_kernel_sizes: {st_spatial_kernel_sizes}")

                param_grid = {
                    'lstm_units': list(map(int, request.POST.get('st_lstm_units', '50').split(','))),
                    'dense_units': list(map(int, request.POST.get('st_dense_units', '50').split(','))),
                    'filters': list(map(int, request.POST.get('st_filters', '64').split(','))),
                    'temporal_kernel_size': list(map(int, request.POST.get('st_temporal_kernel_size', '3').split(','))),
                    'spatial_kernel_sizes': st_spatial_kernel_sizes,
                    'dynamic_attention_units': list(map(int, request.POST.get('st_dynamic_attention_units', '50').split(','))),
                    'epochs': list(map(int, request.POST.get('st_epochs', '10').split(','))),
                    'batch_size': list(map(int, request.POST.get('st_batch_size', '32').split(',')))
                }
            elif request.POST.get('method') == 'traditional':
                param_grid = {
                    'filters': list(map(int, request.POST.get('t_filters', '64').split(','))),  
                    'kernel_size': list(map(int, request.POST.get('t_kernel_size', '3').split(','))), 
                    'lstm_units': list(map(int, request.POST.get('t_lstm_units', '50').split(','))),  
                    'dense_units': list(map(int, request.POST.get('t_dense_units', '50').split(','))),  
                    'dynamic_attention_units': list(map(int, request.POST.get('t_dynamic_attention_units', '50').split(','))),  
                    'epochs': list(map(int, request.POST.get('t_epochs', '10').split(','))),
                    'batch_size': list(map(int, request.POST.get('t_batch_size', '32').split(','))),  
                }

            print(f"Parsed param_grid: {param_grid}")

            # Get test size ratio from form data
            test_size_ratio = float(request.POST.get('test_size_ratio', 0.2))

            # Weights for evaluation calculation
            weights = {
                'w1': float(request.POST.get('w1', 0.25)),
                'w2': float(request.POST.get('w2', 0.25)),
                'w3': float(request.POST.get('w3', 0.25))
            }

            # Train the model using the provided method
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if request.POST.get('method') == 'spatio_temporal':
                future = loop.run_in_executor(None, spatio_temporal_param_combinations, df, target_columns, standard_date_features, time_features, geographic_feature, other_features, param_grid, project_id,  test_size_ratio, weights)
            elif request.POST.get('method') == 'traditional':
                future = loop.run_in_executor(None, traditional_param_combinations, df, target_columns, standard_date_features, other_features, param_grid, project_id, test_size_ratio, weights)

            else:
                return JsonResponse({'status': 'error', 'message': '无效的训练方法'}, status=400)

            result = loop.run_until_complete(future)
            loop.close()

            if isinstance(result, dict):
                for key in ['rmse', 'mae', 'r2', 'adj_r2', 'nse']:
                    if key in result and isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()

            if result['status'] == 'success':
                print("Evaluation completed successfully.")
                return JsonResponse(result)
            else:
                return JsonResponse(result, status=500)

        except Exception as e:
            print(f"Error during evaluation_calculation: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Prepare form options for rendering
    if data_files.exists():
        columns = list(pd.read_json(data_files.first().csv_data).columns)
    else:
        columns = []

    return render(request, 'data_management/evaluation_calculation.html', {
        'project': project,
        'data_files': data_files,
        'columns': columns
    })



@csrf_exempt
def save_evaluation_result(request, project_id):
    try:
        # 获取模型名称
        model_name = request.POST.get('modelName')
        if not model_name:
            return JsonResponse({'status': 'error', 'message': '结果名称不能为空'}, status=400)

        # 构建预测结果文件名
        predicted_file_name = f"{model_name}.csv"
        predicted_csv_file_path = os.path.join('media', str(project_id), 'calculate_result.csv')

        # 读取预测结果文件内容
        with open(predicted_csv_file_path, 'r') as f:
            predicted_csv_content = f.read()

        # 将 CSV 内容转换为 DataFrame
        predicted_df = pd.read_csv(StringIO(predicted_csv_content))

        # 将 DataFrame 转换为 JSON 格式
        predicted_json_content = predicted_df.to_json(orient='records')

        # 创建新的 DataFile 对象并保存到数据库
        project = Project.objects.get(id=project_id)
        new_data_file = DataFile(
            project=project,
            filename=predicted_file_name,
            csv_data=predicted_json_content
        )
        new_data_file.save()

        # 返回成功响应
        return JsonResponse({
            'status': 'success',
            'message': '保存成功！',
            'predicted_file': predicted_file_name
        })
    except Exception as e:
        print(f"Error saving prediction result: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for more details
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



# 设置 TensorFlow 使用单线程以避免多线程问题
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

def data_prediction(request, project_id):
    print("Handling data_prediction request...")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")

    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)
    model_files = ModelFile.objects.filter(project=project)

    if request.method == 'POST':
        try:
            print("Received POST request in data_prediction...")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            model_file_name = request.POST.get('model_file')
            model_file = get_object_or_404(ModelFile, project=project, filename=model_file_name)
            model_data = model_file.h5_data

            # 定义模型文件路径
            model_filename = f'model_{project_id}.h5'
            model_path = os.path.join('media', str(project_id), model_filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # 将字节对象保存为文件
            with open(model_path, 'wb') as f:
                f.write(model_data)

            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            train_data_file_name = request.POST.get('train_data_file')
            if train_data_file_name:
                train_data_file = get_object_or_404(DataFile, project=project, filename=train_data_file_name)
                train_df = pd.read_json(train_data_file.csv_data)
            else:
                raise ValueError("请选择用于标准化的训练文件")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if request.POST.get('method') == 'spatio_temporal':
                future = loop.run_in_executor(None, spatio_temporal_predict_model, df, target_columns, standard_date_features, time_features, geographic_feature, other_features, project_id, train_df)
            elif request.POST.get('method') == 'traditional':
                future = loop.run_in_executor(None, traditional_predict_model, df, target_columns, standard_date_features, other_features, project_id, train_df)
            else:
                return JsonResponse({'status': 'error', 'message': '无效的模型文件'}, status=400)

            result = loop.run_until_complete(future)
            
            loop.close()

            if isinstance(result, dict):
                for key in ['rmse', 'mae', 'r2', 'adj_r2', 'nse']:
                    if key in result and isinstance(result[key], np.ndarray):
                        result[key] = result[key].tolist()

            if result['status'] == 'success':
                print("Prediction saved successfully.")

                # 返回成功响应，前端根据这个状态显示保存结果按钮
                return JsonResponse({
                    'status': 'success',
                    'message': '预测完成！'
                })
            else:
                return JsonResponse(result, status=500)

        except Exception as e:
            print(f"Error during data_prediction: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Prepare form options for rendering
    if data_files.exists():
        columns = list(pd.read_json(data_files.first().csv_data).columns)
    else:
        columns = []

    return render(request, 'data_management/data_prediction.html', {
        'project': project,
        'data_files': data_files,
        'model_files': model_files,
        'columns': columns
    })



@csrf_exempt
def save_prediction_result(request, project_id):
    try:
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

        # 获取预测结果名称
        result_name = request.POST.get('resultName')
        if not result_name:
            return JsonResponse({'status': 'error', 'message': '预测结果名称不能为空'}, status=400)

        # 构建预测结果文件名
        predicted_file_name = f"{result_name}.csv"
        predicted_csv_file_path = os.path.join('media', str(project_id), 'predicted_result.csv')

        # 读取预测结果文件内容
        with open(predicted_csv_file_path, 'r') as f:
            predicted_csv_content = f.read()

        # 将 CSV 内容转换为 DataFrame
        predicted_df = pd.read_csv(StringIO(predicted_csv_content))

        # 将 DataFrame 转换为 JSON 格式
        predicted_json_content = predicted_df.to_json(orient='records')

        # 创建新的 DataFile 对象并保存到数据库
        project = Project.objects.get(id=project_id)
        new_data_file = DataFile(
            project=project,
            filename=predicted_file_name,
            csv_data=predicted_json_content
        )
        new_data_file.save()

        # 返回成功响应
        return JsonResponse({
            'status': 'success',
            'message': '预测结果保存成功！',
            'predicted_file': predicted_file_name
        })
    except Exception as e:
        print(f"Error saving prediction result: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for more details
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)





@csrf_exempt
def save_model(request, project_id):
    try:
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

        features = request.POST.dict()
        target_columns = []
        time_features = []
        standard_date_features = []
        other_features = []
        geographic_feature = []

        for key, value in features.items():
            if key.startswith('features[') and key.endswith(']'):
                column_name = key.split('[')[1].split(']')[0]
                feature_type = value
                if feature_type == 'target_variable':
                    target_columns.append(column_name)
                elif feature_type == 'time_feature':
                    time_features.append(column_name)
                elif feature_type == 'standard_date_feature':
                    standard_date_features.append(column_name)
                elif feature_type == 'geographic_feature':
                    geographic_feature.append(column_name)
                elif feature_type == 'other_feature':
                    other_features.append(column_name)
        
        if not target_columns:
            return JsonResponse({'status': 'error', 'message': '目标变量未选择'}, status=400)
        
        # 获取用户输入的模型名称
        model_name = request.POST.get('model_name')
        if not model_name:
            return JsonResponse({'status': 'error', 'message': '模型名称不能为空'})

        # 构建模型文件名
        model_filename = f'{model_name}.h5'

        model_filename_old = f'model_{project_id}.h5'
        model_path = os.path.join('media', str(project_id), model_filename_old)

        if not os.path.exists(model_path):
            return JsonResponse({'status': 'error', 'message': '模型文件不存在'}, status=404)

        with open(model_path, 'rb') as f:
            h5_data = f.read()

        # 将模型文件保存到数据库
        project = Project.objects.get(id=project_id)
        model_file = ModelFile(
            project=project,
            filename=model_filename,
            h5_data=h5_data
        )
        model_file.save()

        return JsonResponse({'status': 'success', 'message': '结果保存成功'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})







def data_training_dash(request, project_id):
    print("Handling data_training_dash request...")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")

    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    target_columns = []

    # 添加训练计数变量
    training_count = 0

    if request.method == 'POST':
        try:
            print("Received POST request in data_training_dash...")
            print(f"POST data: {request.POST}")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            # Define parameter grid based on user input or default values
            param_grid = {
                'lstm_units': [int(request.POST.get('lstm_units', 50))],
                'dense_units': [int(request.POST.get('dense_units', 50))],
                'filters': [int(request.POST.get('filters', 64))],
                'temporal_kernel_size': [int(request.POST.get('temporal_kernel_size', 3))],
                'spatial_kernel_sizes': [list(map(int, request.POST.get('spatial_kernel_sizes', '[3, 3]').strip('[]').split(',')))],
                'dynamic_attention_units': [int(request.POST.get('dynamic_attention_units', 50))],
                'epochs': [int(request.POST.get('epochs', 10))],
                'batch_size': [int(request.POST.get('batch_size', 32))]
            }

            print(f"Parsed param_grid: {param_grid}")

            # Get test size ratio from form data
            test_size_ratio = float(request.POST.get('test_size_ratio', 0.2))

            # Train the model using the provided method
            model_filename = f'model_{project_id}'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            future = loop.run_in_executor(None, spatio_temporal_train_model, df, target_columns, standard_date_features, time_features, geographic_feature, other_features, param_grid, project_id, model_filename, test_size_ratio)
            result = loop.run_until_complete(future)
            # print(f"Result: {result}")
            loop.close()

            if isinstance(result, dict) and result['status'] == 'success':
                print("Model saved successfully.")
                
                # 处理前一次训练结果的保存
                current_files = [
                    f'{model_filename}_metrics.json',
                    f'{model_filename}_loss.json',
                    f'{model_filename}_predictions.json'
                ]
                previous_files = [
                    f'{model_filename}_previous_metrics.json',
                    f'{model_filename}_previous_loss.json',
                    f'{model_filename}_previous_predictions.json'
                ]
                
                for previous_file in previous_files:
                    file_path = os.path.join('media', str(project_id), previous_file)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as f:
                            f.write('{}')
                
                    else:
                        for current_file, previous_file in zip(current_files, previous_files):
                            current_file_path = os.path.join('media', str(project_id), current_file)
                            previous_file_path = os.path.join('media', str(project_id), previous_file)
                            if os.path.exists(current_file_path):
                                with open(current_file_path, 'r') as current_file_handle:
                                    current_data = current_file_handle.read()
                                with open(previous_file_path, 'w') as previous_file_handle:
                                    previous_file_handle.write(current_data)

                # Generate charts
                plot_loss_web(result['results'], project_id, model_filename, len(target_columns), target_columns)
                plot_metrics_web(result['results'], project_id, model_filename, len(target_columns), target_columns)
                plot_predictions_web(result['y_true'], result['y_pred'], project_id, model_filename, len(target_columns), target_columns)
                plot_feature_importance_web(result['scalers']['X_time'][0], result['scalers']['X_geographical'][0], result['scalers']['X_meteorological'][0],
                                            result['X_time'], result['X_geographical'], result['X_meteorological'],
                                            result['X_time_scaled'], result['X_geographical_scaled'], result['X_meteorological_scaled'],
                                            project_id, model_filename, time_features, geographic_feature, other_features)
                
                features_dict = {
                    'time_feature': time_features,
                    'standard_date_feature': standard_date_features,
                    'geographical_feature': geographic_feature,
                    'other_feature': other_features,
                    'target_variable': target_columns
                }
                
                for key, value in result.items():
                    if isinstance(value, np.float32):
                        result[key] = float(value)
                
                # 更新训练计数变量
                training_count = training_count + 1
                project.save()

                response_data = {
                    'status': 'success',
                    'message': '模型训练并保存成功！',
                    'target_columns': target_columns,
                    'triggered_epoch': result.get('triggered_epoch', None)
                }
                return JsonResponse(response_data, status=200)
            
            else:
                response_data = {
                    'status': 'error',
                    'message': '模型训练失败'
                }
                return JsonResponse(response_data, status=500)

        except Exception as e:
            print(f"Error during data_training_dash: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Prepare form options for rendering
    if data_files.exists():
        columns = list(pd.read_json(data_files.first().csv_data).columns)
    else:
        columns = []

    # Pass target_columns to the template
    context = {
        'project': project,
        'data_files': data_files,
        'columns': columns,
        'target_columns': target_columns  # Add this line
    }
    return render(request, 'data_management/training_dash.html', context)




def get_suggestions(request, project_id):
    print(f"Handling get_suggestions request for project_id={project_id}...")
    try:
        project = get_object_or_404(Project, id=project_id)
        print(f"Project found: {project.name}")
        
        # 构建 suggestions.json 文件路径
        model_filename = f'model_{project_id}'
        suggestions_file_path = os.path.join('media', str(project_id), f'{model_filename}_suggestions.json')
        
        if not os.path.exists(suggestions_file_path):
            print(f"Suggestions file not found: {suggestions_file_path}")
            return JsonResponse({'status': 'error', 'message': 'Suggestions 文件未找到'}, status=404)
        
        # 读取 suggestions.json 文件内容
        with open(suggestions_file_path, 'r', encoding='utf-8') as f:
            suggestions_data = json.load(f)
        
        return JsonResponse({
            'status': 'success',
            'message': 'Suggestions 文件读取成功',
            'suggestions': suggestions_data
        })
    
    except Exception as e:
        print(f"Error retrieving suggestions: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



def delete_previous_file(request, project_id, file_type):
    print(f"Handling delete_previous_files request for project_id={project_id}, file_type={file_type}...")
    try:
        project = get_object_or_404(Project, id=project_id)
        model_filename = f'model_{project_id}'
        previous_files = [
            f'{model_filename}_previous_metrics.json',
            f'{model_filename}_previous_loss.json',
            f'{model_filename}_previous_predictions.json',
            f'{model_filename}_params.json',
        ]
        
        for previous_file in previous_files:
            file_path = os.path.join('media', str(project_id), previous_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted previous file: {file_path}")
        
        return JsonResponse({'status': 'success', 'message': 'Previous files deleted successfully.'})
    except Exception as e:
        print(f"Error deleting previous files: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def get_data_for_charts(request, project_id, data_filename):
    print(f"Handling get_data_for_charts request for project_id={project_id}, data_filename={data_filename}...")
    try:
        project = get_object_or_404(Project, id=project_id)
        print(f"Project found: {project.name}")
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
        print(f"DataFile found: {data_file.filename}")
        df = pd.read_json(data_file.csv_data)

        # Load JSON data for charts
        model_filename = f'model_{project_id}'
        
        radar_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_metrics.json')
        loss_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_loss.json')
        importance_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_feature_importance.json')
        prediction_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_predictions.json')

        previous_radar_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_previous_metrics.json')
        previous_loss_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_previous_loss.json')
        
        previous_prediction_chart_data_path = os.path.join('media', str(project_id), f'{model_filename}_previous_predictions.json')

        radar_chart_data = json.load(open(radar_chart_data_path, 'r')) if os.path.exists(radar_chart_data_path) else {}
        loss_chart_data = json.load(open(loss_chart_data_path, 'r')) if os.path.exists(loss_chart_data_path) else {}
        importance_chart_data = json.load(open(importance_chart_data_path, 'r')) if os.path.exists(importance_chart_data_path) else {}
        prediction_chart_data = json.load(open(prediction_chart_data_path, 'r')) if os.path.exists(prediction_chart_data_path) else {}

        previous_radar_chart_data = json.load(open(previous_radar_chart_data_path, 'r')) if os.path.exists(previous_radar_chart_data_path) else {}
        previous_loss_chart_data = json.load(open(previous_loss_chart_data_path, 'r')) if os.path.exists(previous_loss_chart_data_path) else {}
        
        previous_prediction_chart_data = json.load(open(previous_prediction_chart_data_path, 'r')) if os.path.exists(previous_prediction_chart_data_path) else {}

        return JsonResponse({
            'radarChartData': radar_chart_data,
            'lossChartData': loss_chart_data,
            'importanceChartData': importance_chart_data,
            'predictionChartData': prediction_chart_data,
            'previousRadarChartData': previous_radar_chart_data,
            'previousLossChartData': previous_loss_chart_data,
            'previousPredictionChartData': previous_prediction_chart_data
        })
    except Exception as e:
        print(f"Error retrieving data for charts: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)




def generate_histograms_and_parallel_coordinates(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    target_columns = []

    if request.method == 'POST':
        try:
            print("Received POST request in generate_histograms_and_parallel_coordinates...")
            print(f"POST data: {request.POST}")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            features_dict = {
                'time_feature': time_features,
                'standard_date_feature': standard_date_features,
                'geographical_feature': geographic_feature,
                'other_feature': other_features,
                'target_variable': target_columns
            }

            if standard_date_features:
                date_column = standard_date_features[0]
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.sort_values(by=date_column)
                df['year'] = df[date_column].dt.year
                df['month'] = df[date_column].dt.month
                df['day'] = df[date_column].dt.day
                time_features.extend(['year', 'month', 'day'])

            model_filename = f'model_{project_id}'
            plot_histogram_web(df, features_dict, project_id, model_filename)
            plot_parallel_coordinates_web(df, features_dict, project_id, model_filename)

            # 参数建议逻辑
            # 1. LSTM Units
            # 使用自相关分析来判断目标变量与时间特征之间的时间依赖关系
            target_df = df[target_columns]
            time_df = df[time_features]
            acf_values = {}
            acf_messages = []
            for target_col in target_columns:
                acf_values[target_col] = {}
                for time_col in time_features:
                    # 计算目标变量与时间特征之间的互相关系数
                    acf = target_df[target_col].autocorr(lag=1)  # 使用 lag=1 来计算与时间特征之间的自相关
                    acf_values[target_col][time_col] = acf
                    acf_message = f"{target_col} 和 {time_col} 之间的时间自相关系数: {acf}"
                    acf_messages.append(acf_message)

            # 检查是否存在较强的时间依赖关系
            strong_dependency = any(np.abs(acf) > 0.5 for acfs in acf_values.values() for acf in acfs.values())

            if strong_dependency:
                lstm_units_recommendation = [64, 256]
                lstm_units_recommendation_message = "LSTM_Units_Recommendation: 较大值 (64-256)"
                lstm_units_recommendation_details = "Details: 目标变量与时间特征之间存在较强的时间依赖关系，适合使用较大的LSTM Units来捕捉这些依赖关系。"
                acf_detail_messages = []
                for target_col, acfs in acf_values.items():
                    for time_col, acf in acfs.items():
                        if np.abs(acf) > 0.5:
                            acf_detail_message = f"目标变量 {target_col} 与时间特征 {time_col} 之间存在较强的时间依赖关系 (自相关系数: {acf})"
                            acf_detail_messages.append(acf_detail_message)
            else:
                lstm_units_recommendation = [16, 64]
                lstm_units_recommendation_message = "LSTM_Units_Recommendation: 较小值 (16-64)"
                lstm_units_recommendation_details = "Details: 目标变量与时间特征之间的时间依赖关系较弱，适合使用较小的LSTM Units。"
                acf_detail_messages = []

            # 2. Dense Units
            # 使用互信息或相关系数矩阵来判断非线性关系

            if not other_features:
                raise ValueError("其他特征未选择")

            if not target_columns:
                raise ValueError("目标变量未选择")

            # 确保 other_features 中的列名在 df 中存在
            other_features = [feature for feature in other_features if feature in df.columns]
            if not other_features:
                raise ValueError("选择的其他特征在数据集中不存在")

            # 打印 other_features 和 target_columns 以进行调试
            print(f"other_features: {other_features}")
            print(f"target_columns: {target_columns}")

            dense_units_recommendations = []
            dense_units_recommendation_messages = []
            dense_units_recommendation_details_list = []
            corr_detail_messages_list = []
            corr_detail_messages = []

            for target_col in target_columns:
                # 计算互信息矩阵
                mi_matrix = mutual_info_regression(df[other_features], df[target_col])
                mi_matrix = pd.DataFrame(mi_matrix, index=other_features, columns=[target_col])

                # 打印 mi_matrix 以进行调试
                print(f"mi_matrix for {target_col}:\n{mi_matrix}")

                # 找到互信息矩阵中的最大值
                max_mi = mi_matrix.max().max()
                max_mi_message = f"互信息矩阵中的最大值: {max_mi}"

                # 根据互信息矩阵中的最大值来推荐 Dense Units
                if max_mi > 0.5:
                    dense_units_recommendation = [64, 256]
                    dense_units_recommendation_message = "Dense_Units_Recommendation: 较大值 (64-256)"
                    dense_units_recommendation_details = "Details: 互信息矩阵显示特征之间的关系较为复杂，存在明显的非线性关系，适合使用较大的Dense Units来捕捉这些非线性关系。"
                    
                    for feature in other_features:
                        mi = mi_matrix.loc[feature, target_col]
                        if mi > 0.5:
                            mi_detail_message = f"特征 {feature} 与目标变量 {target_col} 之间存在较强的非线性关系 (互信息: {mi})"
                            corr_detail_messages.append(mi_detail_message)
                else:
                    dense_units_recommendation = [16, 64]
                    dense_units_recommendation_message = "Dense_Units_Recommendation: 较小值 (16-64)"
                    dense_units_recommendation_details = "Details: 互信息矩阵显示特征之间的关系较为简单，不存在明显的非线性关系，适合使用较小的Dense Units。"
                    

                dense_units_recommendations.append(dense_units_recommendation)
                dense_units_recommendation_messages.append(dense_units_recommendation_message)
                dense_units_recommendation_details_list.append(dense_units_recommendation_details)
                corr_detail_messages_list.append(corr_detail_messages)

            # 3. Filters
            # 使用标准差或方差来判断特征分布情况
            std_devs = df.std().to_dict()
            std_devs_message = f"特征的标准差: {std_devs}"
            std_dev_detail_messages = []
            if any(std_dev > 1.0 for std_dev in std_devs.values()):
                filters_recommendation = [64, 128]
                filters_recommendation_message = "Filters_Recommendation: 较大值 (64-128)"
                filters_recommendation_details = "Details: 直方图显示某些特征的分布较为分散，适合使用较大的Filters来捕捉这些特征的细节。"
                for feature, std_dev in std_devs.items():
                    if std_dev > 1.0:
                        std_dev_detail_message = f"特征 {feature} 的标准差较大 (标准差: {std_dev})"
                        std_dev_detail_messages.append(std_dev_detail_message)
            else:
                filters_recommendation = [8, 64]
                filters_recommendation_message = "Filters_Recommendation: 较小值 (8-64)"
                filters_recommendation_details = "Details: 直方图显示所有特征的分布较为集中，适合使用较小的Filters。"
                std_dev_detail_messages = []

            # 4. Temporal Kernel Size
            # 使用傅里叶变换来判断时间特征的周期性
            from scipy.fft import fft
            temporal_features = df[time_features]
            fft_values = np.abs(fft(temporal_features.values, axis=0))
            fft_values_message = f"傅里叶变换值: {fft_values.flatten()}"
            fft_detail_messages = []
            if any(fft_val > 10 for fft_val in fft_values.flatten()):
                temporal_kernel_size_recommendation = [32, 128]
                temporal_kernel_size_recommendation_message = "Temporal_Kernel_Size_Recommendation: 较大值 (32-128)"
                temporal_kernel_size_recommendation_details = "Details: 直方图显示某些时间特征存在明显的周期性变化，适合使用较大的Temporal Kernel Size来捕捉这些周期性变化。"
                for i, feature in enumerate(time_features):
                    if any(fft_values[:, i] > 10):
                        fft_detail_message = f"时间特征 {feature} 存在明显的周期性 (傅里叶变换值大于10)"
                        fft_detail_messages.append(fft_detail_message)
            else:
                temporal_kernel_size_recommendation = [2, 32]
                temporal_kernel_size_recommendation_message = "Temporal_Kernel_Size_Recommendation: 较小值 (2-32)"
                temporal_kernel_size_recommendation_details = "Details: 直方图显示时间特征不存在明显的周期性变化，适合使用较小的Temporal Kernel Size。"
                fft_detail_messages = []

            # 5. Spatial Kernel Sizes
            # 使用聚类分析来判断地理特征的空间聚集性
            from sklearn.cluster import KMeans
            geo_features = df[geographic_feature]
            cluster_counts_messages = []
            if geo_features.shape[1] > 0:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(geo_features)
                cluster_labels = kmeans.labels_
                num_clusters = len(set(cluster_labels))
                num_clusters_message = f"地理特征的聚类数量: {num_clusters}"
                if num_clusters > 1:
                    spatial_kernel_sizes_recommendation = [32, 128]
                    spatial_kernel_sizes_recommendation_message = "Spatial_Kernel_Sizes_Recommendation: 较大值 (32-128)"
                    spatial_kernel_sizes_recommendation_details = "Details: 直方图显示地理特征存在明显的空间聚集性，适合使用较大的Spatial Kernel Sizes来捕捉这些空间聚集性。"
                    for i, feature in enumerate(geographic_feature):
                        cluster_counts = pd.Series(cluster_labels).value_counts().to_dict()
                        cluster_counts_message = f"地理特征 {feature} 的聚类分布: {cluster_counts}"
                        cluster_counts_messages.append(cluster_counts_message)
                else:
                    spatial_kernel_sizes_recommendation = [2, 32]
                    spatial_kernel_sizes_recommendation_message = "Spatial_Kernel_Sizes_Recommendation: 较小值 (2-32)"
                    spatial_kernel_sizes_recommendation_details = "Details: 直方图显示地理特征不存在明显的空间聚集性，适合使用较小的Spatial Kernel Sizes。"
                    cluster_counts_messages = []
            else:
                spatial_kernel_sizes_recommendation = [2, 32]
                spatial_kernel_sizes_recommendation_message = "Spatial_Kernel_Sizes_Recommendation: 较小值 (2-32)"
                spatial_kernel_sizes_recommendation_details = "Details: 直方图显示没有提供地理特征，适合使用较小的Spatial Kernel Sizes。"
                cluster_counts_messages = []

            # 6. Dynamic Attention Units
            # 使用特征重要性分析来判断特征的重要性随时间的变化
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=42)
            model.fit(df[other_features], df[target_columns])
            importances = model.feature_importances_
            importances_message = f"Feature importances: {importances.tolist()}"  # 将 importances 数组转换为列表
            importance_detail_messages = []
            if any(importance > 0.2 for importance in importances):
                dynamic_attention_units_recommendation = [64, 256]
                dynamic_attention_units_recommendation_message = "Dynamic_Attention_Units_Recommendation: 较大值 (64-256)"
                dynamic_attention_units_recommendation_details = "Details: 使用特征重要性分析发现某些特征的重要性随时间的变化较高，适合使用较大的Dynamic Attention Units来捕捉这些特征的重要性。"
                for i, feature in enumerate(other_features):
                    if importances[i] > 0.2:
                        importance_detail_message = f"特征 {feature} 的重要性较高 (重要性: {importances[i]})"
                        importance_detail_messages.append(importance_detail_message)
            else:
                dynamic_attention_units_recommendation = [16, 64]
                dynamic_attention_units_recommendation_message = "Dynamic_Attention_Units_Recommendation: 较小值 (16-64)"
                dynamic_attention_units_recommendation_details = "Details: 使用特征重要性分析发现所有特征的重要性随时间的变化较低，适合使用较小的Dynamic Attention Units。"
                importance_detail_messages = []

            # 7. Epochs
            # 根据数据量和特征数量来建议Epochs
            data_size = len(df)
            num_features = len(df.columns)
            data_size_message = f"数据量: {data_size}"
            num_features_message = f"特征数量: {num_features}"
            if data_size < 500:
                epochs_recommendation = [50, 200]
                epochs_recommendation_message = "Epochs_Recommendation: 较大值 (50-300)"
                epochs_recommendation_details = "EDetails: 因为数据量小不容易训练，适合使用较大的Epochs来增加拟合度，但不建议用少于100的数据训练。"
            elif data_size < 1000:
                epochs_recommendation = [30, 100]
                epochs_recommendation_message = "Epochs_Recommendation: 适中值 (30-100)"
                epochs_recommendation_details = "Details: 因为数据量适中或特征数量适中，适合使用适中的Epochs。"
            elif data_size < 3000:
                epochs_recommendation = [20, 80]
                epochs_recommendation_message = "Epochs_Recommendation: 适中值 (30-100)"
                epochs_recommendation_details = "Details: 因为数据量适中或特征数量适中，适合使用适中的Epochs。"
            elif data_size < 5000:
                epochs_recommendation = [10, 70]
                epochs_recommendation_message = "Epochs_Recommendation: 适中值 (10-100)"
                epochs_recommendation_details = "Details: 因为数据量适中或特征数量适中，适合使用适中的Epochs。"
            else:
                epochs_recommendation = [5, 50]
                epochs_recommendation_message = "Epochs_Recommendation: 较小值 (5-50)"
                epochs_recommendation_details = "Details: 因为数据量较大或特征数量较多，很容易训练得到良好的拟合度，当然Epochs可以越大越好直到触发早停，不过多出的大量时间不会为拟合度带来太多的优化。"

            # 8. Batch Size
            # 根据数据量来建议Batch Size
            batch_size_recommendation_message = ""
            batch_size_recommendation_details = ""
            if data_size < 1000:
                batch_size_recommendation = [8, 64]
                batch_size_recommendation_message = "Batch_Size_Recommendation: 较小值 (8-64)"
                batch_size_recommendation_details = "Details: 直方图和平行坐标图显示数据量较小，适合使用较小的Batch Size来提高训练效率。"
            elif data_size < 5000:
                batch_size_recommendation = [16, 128]
                batch_size_recommendation_message = "Batch_Size_Recommendation: 适中值 (16-128)"
                batch_size_recommendation_details = "Details: 直方图和平行坐标图显示数据量适中，适合使用适中的Batch Size。"
            else:
                batch_size_recommendation = [32, 256]
                batch_size_recommendation_message = "Batch_Size_Recommendation: 较大值 (32-256)"
                batch_size_recommendation_details = "Details: 直方图和平行坐标图显示数据量较大，适合使用较大的Batch Size来提高训练效率。"

            response_data = {
                'status': 'success',
                'message': '直方图和平行坐标图数据生成成功！',
                'target_columns': target_columns,
                'lstm_units_recommendation': lstm_units_recommendation,
                'dense_units_recommendation': dense_units_recommendation,
                'filters_recommendation': filters_recommendation,
                'temporal_kernel_size_recommendation': temporal_kernel_size_recommendation,
                'spatial_kernel_sizes_recommendation': spatial_kernel_sizes_recommendation,
                'dynamic_attention_units_recommendation': dynamic_attention_units_recommendation,
                'epochs_recommendation': epochs_recommendation,
                'batch_size_recommendation': batch_size_recommendation,
                'lstm_units_recommendation_message': lstm_units_recommendation_message,
                'lstm_units_recommendation_details': lstm_units_recommendation_details,
                'acf_detail_messages':acf_detail_messages,
                'dense_units_recommendation_message': dense_units_recommendation_message,
                'dense_units_recommendation_details': dense_units_recommendation_details,
                'corr_detail_messages':corr_detail_messages,
                'filters_recommendation_message': filters_recommendation_message,
                'filters_recommendation_details': filters_recommendation_details,
                'std_dev_detail_messages':std_dev_detail_messages,
                'temporal_kernel_size_recommendation_message': temporal_kernel_size_recommendation_message,
                'temporal_kernel_size_recommendation_details': temporal_kernel_size_recommendation_details,
                'fft_detail_messages':fft_detail_messages,
                'spatial_kernel_sizes_recommendation_message': spatial_kernel_sizes_recommendation_message,
                'spatial_kernel_sizes_recommendation_details': spatial_kernel_sizes_recommendation_details,
                'cluster_counts_messages':cluster_counts_messages,
                'dynamic_attention_units_recommendation_message': dynamic_attention_units_recommendation_message,
                'dynamic_attention_units_recommendation_details': dynamic_attention_units_recommendation_details,
                'importance_detail_messages':importance_detail_messages,
                'epochs_recommendation_message': epochs_recommendation_message,
                'epochs_recommendation_details': epochs_recommendation_details,
                'batch_size_recommendation_message': batch_size_recommendation_message,
                'batch_size_recommendation_details': batch_size_recommendation_details,
                'num_clusters_message': num_clusters_message,
                'data_size_message': data_size_message,
                'num_features_message': num_features_message,
            }
            
            # 移除所有字符串中的换行符
            response_data = {k: v.replace('\n', '') if isinstance(v, str) else v for k, v in response_data.items()}


            print('response_data:', response_data)
            return JsonResponse(response_data)
        except Exception as e:
            print(f"Error during generate_histograms_and_parallel_coordinates: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)




def get_histogram_and_parallel_coordinates_data(request, project_id, data_filename):
    print(f"Handling get_histogram_and_parallel_coordinates_data request for project_id={project_id}, data_filename={data_filename}...")
    try:
        project = get_object_or_404(Project, id=project_id)
        print(f"Project found: {project.name}")
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
        print(f"DataFile found: {data_file.filename}")

        model_filename = f'model_{project_id}'
        histogram_data_path = os.path.join('media', str(project_id), f'{model_filename}_histograms.json')
        histogram_2d_data_path = os.path.join('media', str(project_id), f'{model_filename}_2d_histograms.json')
        histogram_3d_data_path = os.path.join('media', str(project_id), f'{model_filename}_3d_histograms.json')
        parallel_coordinates_data_path = os.path.join('media', str(project_id), f'{model_filename}_parallel_coordinates.json')

        
        histogram_data = json.load(open(histogram_data_path, 'r')) if os.path.exists(histogram_data_path) else {}
        histogram_2d_data = json.load(open(histogram_2d_data_path, 'r')) if os.path.exists(histogram_2d_data_path) else {}
        histogram_3d_data = json.load(open(histogram_3d_data_path, 'r')) if os.path.exists(histogram_3d_data_path) else {}
        parallel_coordinates_data = json.load(open(parallel_coordinates_data_path, 'r')) if os.path.exists(parallel_coordinates_data_path) else {}

     

        return JsonResponse({
            'histogramData': histogram_data,
            'histogram2dData': histogram_2d_data,
            'histogram3dData': histogram_3d_data,
            'parallelCoordinatesData': parallel_coordinates_data
        })
    except Exception as e:
        print(f"Error retrieving data for histograms and parallel coordinates: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)





def get_data_size(request, project_id, data_filename):
    try:
        project = get_object_or_404(Project, id=project_id)
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
        df = pd.read_json(data_file.csv_data)
        data_size = len(df)
        return JsonResponse({'data_size': data_size})
    except Exception as e:
        print(f"Error retrieving data size: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



# 全局变量来存储训练线程和事件
training_threads = {}

@csrf_exempt
def stop_hyperparameter_tuning(request, project_id):
    try:
        print(f"Handling stop_hyperparameter_tuning request for project_id={project_id}...")
        project = get_object_or_404(Project, id=project_id)

        # 检查是否有正在进行的训练
        if project_id in training_threads:
            # 设置停止事件
            training_threads[project_id]['stop_event'].set()
            print(f"Training stopped for project_id={project_id}.")


            # 读取 best_result.json 文件
            model_filename = f'model_{project_id}'
            best_result_path = os.path.join('media', str(project_id), f'{model_filename}_best_result.json')
            if os.path.exists(best_result_path):
                with open(best_result_path, 'r') as f:
                    best_result_data = json.load(f)
                return JsonResponse({'status': 'success', 'message': '调优已停止', 'best_result': best_result_data})
            else:
                return JsonResponse({'status': 'success', 'message': '调优已停止'})
        else:
            print(f"No active training found for project_id={project_id}.")
            return JsonResponse({'status': 'error', 'message': '没有正在进行的调优'})
    except Exception as e:
        print(f"Error stopping hyperparameter tuning: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def hyperparameter_training(request, project_id):
    print("Handling hyperparameter_training request...")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")

    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    if request.method == 'POST':
        try:
            print("Received POST request in hyperparameter_training...")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            # Define parameter grid based on user input or default values
            param_grid = {}
            if request.POST.get('method') == 'spatio_temporal':
                st_spatial_kernel_sizes_str = request.POST.get('st_spatial_kernel_sizes', '[[3, 3], [4, 4]]')
                st_spatial_kernel_sizes = eval(st_spatial_kernel_sizes_str) 

                print(f"Received st_spatial_kernel_sizes: {st_spatial_kernel_sizes}")

                param_grid = {
                    'lstm_units': list(map(int, request.POST.get('st_lstm_units', '50').split(','))),
                    'dense_units': list(map(int, request.POST.get('st_dense_units', '50').split(','))),
                    'filters': list(map(int, request.POST.get('st_filters', '64').split(','))),
                    'temporal_kernel_size': list(map(int, request.POST.get('st_temporal_kernel_size', '3').split(','))),
                    'spatial_kernel_sizes': st_spatial_kernel_sizes,
                    'dynamic_attention_units': list(map(int, request.POST.get('st_dynamic_attention_units', '50').split(','))),
                    'epochs': list(map(int, request.POST.get('st_epochs', '10').split(','))),
                    'batch_size': list(map(int, request.POST.get('st_batch_size', '32').split(',')))
                }
            elif request.POST.get('method') == 'traditional':
                param_grid = {
                    'filters': list(map(int, request.POST.get('t_filters', '64').split(','))),  
                    'kernel_size': list(map(int, request.POST.get('t_kernel_size', '3').split(','))), 
                    'lstm_units': list(map(int, request.POST.get('t_lstm_units', '50').split(','))),  
                    'dense_units': list(map(int, request.POST.get('t_dense_units', '50').split(','))),  
                    'dynamic_attention_units': list(map(int, request.POST.get('t_dynamic_attention_units', '50').split(','))),  
                    'epochs': list(map(int, request.POST.get('t_epochs', '10').split(','))),
                    'batch_size': list(map(int, request.POST.get('t_batch_size', '32').split(','))),  
                }

            print(f"Parsed param_grid: {param_grid}")

            # Get test size ratio from form data
            test_size_ratio = float(request.POST.get('test_size_ratio', 0.2))

            # Get n_calls from form data
            n_calls = int(request.POST.get('n_calls', 10))  # 默认值为10

            # Get weights from form data
            weights = {
                'w1': float(request.POST.get('w1', 0.25)),
                'w2': float(request.POST.get('w2', 0.25)),
                'w3': float(request.POST.get('w3', 0.5))
            }

            # 创建停止事件
            stop_event = threading.Event()

            # 定义训练函数
            def train_function():
                try:
                    if request.POST.get('method') == 'spatio_temporal':
                        result = train_with_bayesian_optimization(df, target_columns, standard_date_features, time_features, geographic_feature, other_features, project_id, f'model_{project_id}', test_size_ratio, stop_event, n_calls=n_calls, weights=weights)
                    elif request.POST.get('method') == 'traditional':
                        result = traditional_train_with_bayesian_optimization(df, target_columns, standard_date_features, other_features, param_grid, project_id, f'model_{project_id}', test_size_ratio, stop_event, n_calls=n_calls, weights=weights)
                    else:
                        return JsonResponse({'status': 'error', 'message': '无效的模型文件'}, status=400)

                    if result['status'] == 'success':
                        print("Hyperparameter training completed successfully.")
                    else:
                        print("Hyperparameter training failed.")
                except Exception as e:
                    print(f"Error during hyperparameter_training: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for more details

            # 启动训练线程
            training_thread = threading.Thread(target=train_function)
            training_thread.start()

            # 存储线程和停止事件
            training_threads[project_id] = {'thread': training_thread, 'stop_event': stop_event}

            return JsonResponse({'status': 'success', 'message': '调优已开始'})

        except Exception as e:
            print(f"Error during hyperparameter_training: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for more details
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # Prepare form options for rendering
    if data_files.exists():
        columns = list(pd.read_json(data_files.first().csv_data).columns)
    else:
        columns = []

    return render(request, 'data_management/hyperparameter_tuning.html', {
        'project': project,
        'data_files': data_files,
        'columns': columns
    })
        



@csrf_exempt
def save_hyperparameter_result(request, project_id):
    try:
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

        features = request.POST.dict()
        target_columns = []
        time_features = []
        standard_date_features = []
        other_features = []
        geographic_feature = []

        for key, value in features.items():
            if key.startswith('features[') and key.endswith(']'):
                column_name = key.split('[')[1].split(']')[0]
                feature_type = value
                if feature_type == 'target_variable':
                    target_columns.append(column_name)
                elif feature_type == 'time_feature':
                    time_features.append(column_name)
                elif feature_type == 'standard_date_feature':
                    standard_date_features.append(column_name)
                elif feature_type == 'geographic_feature':
                    geographic_feature.append(column_name)
                elif feature_type == 'other_feature':
                    other_features.append(column_name)
        
        if not target_columns:
            return JsonResponse({'status': 'error', 'message': '目标变量未选择'}, status=400)
        
        # 获取用户输入的模型名称
        model_name = request.POST.get('model_name')
        if not model_name:
            return JsonResponse({'status': 'error', 'message': '模型名称不能为空'})

        # 构建模型文件名
        model_filename = f'{model_name}.h5'
        model_path = os.path.join('media', str(project_id), model_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 读取 best_model.h5 文件内容
        best_model_path = os.path.join('media', str(project_id), 'best_model.h5')
        if not os.path.exists(best_model_path):
            return JsonResponse({'status': 'error', 'message': '模型文件不存在'}, status=404)

        with open(best_model_path, 'rb') as f:
            h5_data = f.read()

        # 将模型文件保存到数据库
        project = Project.objects.get(id=project_id)
        model_file = ModelFile(
            project=project,
            filename=model_filename,
            h5_data=h5_data
        )
        model_file.save()

        return JsonResponse({'status': 'success', 'message': '结果保存成功'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})





# views.py
@csrf_exempt
def bayesian_optimization(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    data_files = DataFile.objects.filter(project=project)

    if request.method == 'POST':
        try:
            print("Received POST request in bayesian_optimization...")

            data_file_name = request.POST.get('data_file')
            data_file = get_object_or_404(DataFile, project=project, filename=data_file_name)
            df = pd.read_json(data_file.csv_data)

            # Retrieve selected features and target variables from form data
            features = request.POST.dict()
            target_columns = []
            time_features = []
            standard_date_features = []
            other_features = []
            geographic_feature = []

            for key, value in features.items():
                if key.startswith('features[') and key.endswith(']'):
                    column_name = key.split('[')[1].split(']')[0]
                    feature_type = value
                    if feature_type == 'target_variable':
                        target_columns.append(column_name)
                    elif feature_type == 'time_feature':
                        time_features.append(column_name)
                    elif feature_type == 'standard_date_feature':
                        standard_date_features.append(column_name)
                    elif feature_type == 'geographic_feature':
                        geographic_feature.append(column_name)
                    elif feature_type == 'other_feature':
                        other_features.append(column_name)

            if not target_columns:
                raise ValueError("目标变量未选择")

            if not standard_date_features and time_features:
                raise ValueError("时间特征或标准日期特征未选择")

            # Define parameter grid based on user input or default values
            param_grid = {
                'lstm_units': list(map(int, request.POST.get('lstm_units', '50').split(','))),
                'dense_units': list(map(int, request.POST.get('dense_units', '50').split(','))),
                'filters': list(map(int, request.POST.get('filters', '64').split(','))),
                'temporal_kernel_size': list(map(int, request.POST.get('temporal_kernel_size', '3').split(','))),
                'spatial_kernel_sizes': [list(map(int, request.POST.get('spatial_kernel_sizes', '[3, 3]').strip('[]').split(',')))],
                'dynamic_attention_units': list(map(int, request.POST.get('dynamic_attention_units', '50').split(','))),
                'epochs': list(map(int, request.POST.get('epochs', '10').split(','))),
                'batch_size': list(map(int, request.POST.get('batch_size', '32').split(',')))
            }

            print(f"Parsed param_grid: {param_grid}")

            # Get test size ratio from form data
            test_size_ratio = float(request.POST.get('test_size_ratio', 0.2))

            # Get n_calls from form data
            n_calls = int(request.POST.get('n_calls', 10))  # 默认值为10

            weights = {
                'w1': float(request.POST.get('w1', 0.25)),
                'w2': float(request.POST.get('w2', 0.25)),
                'w3': float(request.POST.get('w3', 0.5))
            }

            # Perform Bayesian Optimization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            future = loop.run_in_executor(None, train_with_bayesian, df, target_columns, standard_date_features, time_features, geographic_feature, other_features, project_id, test_size_ratio, n_calls, weights)
            result = loop.run_until_complete(future)
            loop.close()

            if result['status'] == 'success':
                return JsonResponse({
                    'status': 'success',
                    'message': '贝叶斯优化完成',
                    'best_params': result.get('best_params', {}),
                    'score': result.get('score', 999999)
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': result.get('message', '贝叶斯优化失败')
                }, status=500)

        except Exception as e:
            print(f"Error during bayesian_optimization: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        # Handle non-POST requests
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


def get_columns(request, project_id, data_filename):
    print(f"Handling get_columns request for project_id={project_id}, data_filename={data_filename}...")
    try:
        project = get_object_or_404(Project, id=project_id)
        print(f"Project found: {project.name}")
        data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
        print(f"DataFile found: {data_file.filename}")
        df = pd.read_json(data_file.csv_data)
        columns = list(df.columns)
        print(f"Columns retrieved: {columns}")
        return JsonResponse({'columns': columns})
    except Exception as e:
        print(f"Error retrieving columns: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



def visualization_results(request, project_id):
    print("Handling visualization_results request...")
    image_paths = [
        os.path.join('media', str(project_id), 'model_' + str(project_id) + '_loss.png'),
        os.path.join('media', str(project_id), 'model_' + str(project_id) + '_metrics_comparison.png'),
        os.path.join('media', str(project_id), 'model_' + str(project_id) + '_predictions.png'),  
        os.path.join('media', str(project_id), 'model_' + str(project_id) + '_feature_importance.png')
    ]
    images = [{'path': img_path, 'name': os.path.basename(img_path)} for img_path in image_paths if os.path.exists(img_path)]
    context = {
        'project_id': project_id,
        'images': images
    }
    return render(request, 'data_management/visualization_results.html', context)

def download_pdf(request, project_id):
    print("Handling download_pdf request...")
    project = get_object_or_404(Project, id=project_id)
    
    # Construct the path to the generated PDF report
    model_filename = f'model_{project_id}'
    pdf_path = os.path.join('media', str(project_id), f'{model_filename}_report.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return JsonResponse({'status': 'error', 'message': 'PDF文件未找到'}, status=404)

    # Read the PDF file and create an HTTP response
    with open(pdf_path, 'rb') as pdf_file:
        response = HttpResponse(pdf_file.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="training_results_{project.name}.pdf"'
        return response


def download_data(request, project_id, data_filename):
    print("Handling download_data request...")
    project = get_object_or_404(Project, id=project_id)
    data_file = get_object_or_404(DataFile, project=project, filename=data_filename)
    
    try:
        df = pd.read_json(data_file.csv_data)
        csv_content = df.to_csv(index=False)

        response = HttpResponse(csv_content, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{data_file.filename}"'
        return response
    except Exception as e:
        print(f"Error downloading file {data_file.filename}: {str(e)}")
        return JsonResponse({'status': 'error', 'message': '文件未找到'}, status=404)

def download_model(request, project_id, model_filename):
    print("Handling download_model request...")
    project = get_object_or_404(Project, id=project_id)
    model_file = get_object_or_404(ModelFile, project=project, filename=model_filename)

    if not model_file.h5_data:
        print(f"No data found for file: {model_file.filename}")
        return JsonResponse({'status': 'error', 'message': '文件数据未找到'}, status=404)

    response = HttpResponse(model_file.h5_data, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{model_file.filename}"'
    return response





def delete_project_and_files(project_id):
    print("Handling delete_project_and_files request...")
    try:
        project = get_object_or_404(Project, id=project_id)
        
        # Delete all related DataFiles
        data_files = DataFile.objects.filter(project=project)
        for data_file in data_files:
            file_path = os.path.join('media', str(project.id), data_file.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted data file: {file_path}")
            data_file.delete()
            print(f"Deleted DataFile record: {data_file.filename}")

        # Delete all related ModelFiles
        model_files = ModelFile.objects.filter(project=project)
        for model_file in model_files:
            file_path = os.path.join('media', str(project.id), model_file.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted model file: {file_path}")
            model_file.delete()
            print(f"Deleted ModelFile record: {model_file.filename}")

        # Delete the entire project media directory
        project_media_dir = os.path.join('media', str(project.id))
        if os.path.exists(project_media_dir):
            shutil.rmtree(project_media_dir)
            print(f"Deleted directory: {project_media_dir}")

        # Delete the project itself
        project.delete()
        print(f"Deleted project: {project.name}")

        return redirect('personal_projects')
    except Exception as e:
        print(f"Error deleting project and files: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    

