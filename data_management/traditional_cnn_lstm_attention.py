import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Attention, LayerNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping,Callback
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import traceback
import asyncio
from channels.layers import get_channel_layer
from skopt.space import Integer, Categorical
from asgiref.sync import async_to_sync
import tensorflow as tf
from .training_utils import evaluate_models

def plot_loss(results, project_id, model_filename, num_outputs, target_columns):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    
    for i, result in enumerate(results):
        history = result['history']
        for j in range(num_outputs):
            train_key = f'output_{j}_loss' if num_outputs > 1 else 'loss'
            val_key = f'val_output_{j}_loss' if num_outputs > 1 else 'val_loss'
            
            plt.plot(history['loss'][train_key], label=f'Train Loss ({target_columns[j]})', color=colors[j*2 % len(colors)])
            plt.plot(history['val_loss'][val_key], label=f'Val Loss ({target_columns[j]})', linestyle='--', color=colors[j*2+1 % len(colors)])
    
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join('media', str(project_id), f'{model_filename}_loss.png')
    plt.savefig(plot_path)
    plt.close()

def plot_metrics(results, project_id, model_filename, num_outputs, target_columns):
    metrics = ['rmse', 'mae', 'r2', 'adj_r2', 'nse']
    all_rmse = [[] for _ in range(num_outputs)]
    all_mae = [[] for _ in range(num_outputs)]
    all_r2 = [[] for _ in range(num_outputs)]
    all_adj_r2 = [[] for _ in range(num_outputs)]
    all_nse = [[] for _ in range(num_outputs)]

    for result in results:
        rmse_values = np.array(result['rmse'])
        mae_values = np.array(result['mae'])
        r2_values = np.array(result['r2'])
        adj_r2_values = np.array(result['adj_r2'])
        nse_values = np.array(result['nse'])

        for j in range(num_outputs):
            all_rmse[j].append(rmse_values[j])
            all_mae[j].append(mae_values[j])
            all_r2[j].append(r2_values[j])
            all_adj_r2[j].append(adj_r2_values[j])
            all_nse[j].append(nse_values[j])

    labels = [f'Model {i}' for i in range(len(all_rmse[0]))]
    width = 0.15
    x = np.arange(len(labels))

    fig, axs = plt.subplots(num_outputs, figsize=(12, 8 * num_outputs))
    if num_outputs == 1:
        axs = [axs]

    for j in range(num_outputs):
        rects1 = axs[j].bar(x - 2*width, all_rmse[j], width, label='RMSE')
        rects2 = axs[j].bar(x - width, all_mae[j], width, label='MAE')
        rects3 = axs[j].bar(x, all_r2[j], width, label='R2')
        rects4 = axs[j].bar(x + width, all_adj_r2[j], width, label='Adj R2')
        rects5 = axs[j].bar(x + 2*width, all_nse[j], width, label='NSE')

        axs[j].set_xlabel('Models')
        axs[j].set_ylabel('Metrics')
        axs[j].set_title(f'Model Metrics Comparison for {target_columns[j]}')
        axs[j].set_xticks(x)
        axs[j].set_xticklabels(labels)
        axs[j].legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                axs[j].annotate('{:.2f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        autolabel(rects5)

    plot_path = os.path.join('media', str(project_id), f'{model_filename}_metrics_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def plot_predictions(y_true, y_pred, project_id, model_filename, num_outputs, target_columns):
    fig, axs = plt.subplots(num_outputs, figsize=(10, 5 * num_outputs))
    if num_outputs == 1:
        axs = [axs]

    for j in range(num_outputs):
        if num_outputs > 1:
            axs[j].plot(y_true[:, j], label='True Values')
            axs[j].plot(y_pred[j].flatten(), label='Predicted Values')  # 修改这里
        else:
            axs[j].plot(y_true.flatten(), label='True Values')
            axs[j].plot(y_pred.flatten(), label='Predicted Values')
        axs[j].set_xlabel('Samples')
        axs[j].set_ylabel('Values')
        axs[j].legend()
        axs[j].set_title(f'True vs Predicted Values for {target_columns[j]}')

    plot_path = os.path.join('media', str(project_id), f'{model_filename}_predictions.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_feature_importance(scaler, X_meteorological, X_meteorological_scaled, project_id, model_filename, meteorological_features):
    # Check if feature_names_in_ exists
    if hasattr(scaler, 'feature_names_in_'):
        original_feature_names = scaler.feature_names_in_
    else:
        original_feature_names = meteorological_features  # Use provided feature names

    scaled_feature_means = X_meteorological_scaled.mean(axis=0)
    feature_importances = np.argsort(scaled_feature_means)[::-1]  # Ensure feature_importances is an array

    print(f"Original Feature Names: {original_feature_names}")
    print(f"Scaled Feature Means: {scaled_feature_means}")
    print(f"Feature Importances: {feature_importances}")

    plt.figure(figsize=(12, 8))
    plt.bar(np.array(original_feature_names)[feature_importances], scaled_feature_means[feature_importances])  # Use np.array to ensure correct indexing
    plt.title('Feature Importances based on Mean Scaled Values')
    plt.xlabel('Features')
    plt.ylabel('Mean Scaled Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plot_path = os.path.join('media', str(project_id), f'{model_filename}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()

def combine_plots_to_pdf(project_id, model_filename, num_outputs, has_meteorological_features, X_meteorological, X_meteorological_scaled, scaler, meteorological_features):
    pdf = FPDF()
    images = [
        f'{model_filename}_loss.png',
        f'{model_filename}_metrics_comparison.png',
        f'{model_filename}_predictions.png'
    ]
    if has_meteorological_features and X_meteorological.shape[1] > 0:
        images.append(f'{model_filename}_feature_importance.png')
    
    image_paths = [os.path.join('media', str(project_id), img) for img in images]
    for image_path in image_paths:
        if os.path.exists(image_path):
            pdf.add_page()
            pdf.image(image_path, x=10, y=8, w=190)
        else:
            print(f"Image not found: {image_path}")
    pdf_path = os.path.join('media', str(project_id), f'{model_filename}_report.pdf')
    pdf.output(pdf_path)

# 自定义注意力机制层
class CustomAttention(tf.keras.layers.Attention):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(CustomAttention, self).get_config()
        return {**base_config}


def build_model(input_shape, num_outputs,filters,kernel_size,lstm_units,dense_units,dynamic_attention_units):
    inputs = Input(shape=input_shape)
    
    # 卷积层
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(conv1)
    
    # LSTM 层
    lstm1 = LSTM(lstm_units, return_sequences=True)(conv2)
    lstm2 = LSTM(lstm_units, return_sequences=False)(lstm1)
    
    # 注意力机制
    query = Dense(dense_units)(tf.expand_dims(lstm2, 1))  
    value = Dense(dense_units)(tf.expand_dims(lstm2, 1))
    attention = CustomAttention()([query, value])
    attention = Flatten()(attention)  
    
    # 层归一化
    norm_attention = LayerNormalization()(attention)
    
    # 全连接层
    dense1 = Dense(dynamic_attention_units, activation='relu')(norm_attention)
    
    # 输出层
    if num_outputs > 1:
        outputs = [Dense(1, name=f'output_{i}') for i in range(num_outputs)]
        model = Model(inputs=inputs, outputs=[out(dense1) for out in outputs])
        loss_dict = {f'output_{i}': 'mean_squared_error' for i in range(num_outputs)}
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=loss_dict)
    else:
        outputs = Dense(1, name='output_0')(dense1)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='mean_squared_error')
    
    return model




def train_model(data, target_columns, standard_date_features, meteorological_features, param_grid, project_id, model_filename, test_size_ratio=0.2):
    try:
        print("Starting train_model...")
        print(f"Target columns: {target_columns}")
        print(f"Standard date features: {standard_date_features}")
        print(f"Meteorological features: {meteorological_features}")
        print(f"Param grid: {param_grid}")
        print(f"Test size ratio: {test_size_ratio}")

        csv_file_path = os.path.join('media', str(project_id), 'data.csv')
        data.to_csv(csv_file_path, index=False)
        data = pd.read_csv(csv_file_path)

        if standard_date_features:
            date_column = standard_date_features[0]

            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            data['year'] = data[date_column].dt.year
            data['month'] = data[date_column].dt.month
            data['day'] = data[date_column].dt.day
            meteorological_features.extend(['year', 'month', 'day'])

        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)  

        print(f"X_meteorological shape: {X_meteorological.shape}")
        print(f"y shape: {y.shape}")

        if X_meteorological.shape[1] > 0:
            scaler = StandardScaler()
            X_meteorological_scaled = scaler.fit_transform(X_meteorological)
        else:
            X_meteorological_scaled = X_meteorological

        # 数据准备
        X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
           X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)


        results = []
        
        param_dict = next(iter(product(*param_grid.values())))
        param_keys = list(param_grid.keys())
        
        epochs = param_dict[param_keys.index('epochs')]
        batch_size = param_dict[param_keys.index('batch_size')]

        param_dict={}
        for params in list(product(*param_grid.values())):
            param_dict = dict(zip(param_grid.keys(), params))
            epochs = param_dict.pop('epochs')
            batch_size = param_dict.pop('batch_size')
            print('param_dict', param_dict)
        

        total_iterations = epochs
        current_iteration = 0

        channel_layer = get_channel_layer()

        
        class ProgressCallback(Callback):
            def __init__(self, total_iterations, epochs, project_id):
                super().__init__()
                self.total_iterations = total_iterations
                self.epochs = epochs
                self.project_id = project_id
                self.current_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                nonlocal current_iteration
                current_iteration += 1
                progress = int((current_iteration / self.total_iterations) * 100)
                print(f"Epoch {epoch + 1}/{self.epochs}, Iteration {current_iteration}/{self.total_iterations}, Progress: {progress}%")
                try:
                    async_to_sync(channel_layer.group_send)(
                        f'training_progress_{self.project_id}',
                        {
                            'type': 'send_progress',
                            'progress': progress
                        }
                    )
                except Exception as e:
                    print(f"Error sending progress: {str(e)}")

            def get_config(self):
                config = super().get_config()
                config.update({
                    'total_iterations': self.total_iterations,
                    'epochs': self.epochs,
                    'project_id': self.project_id
                })
                return config

        # 自定义回调类
        class CustomCallback(Callback):
            def __init__(self, target_columns):
                super(CustomCallback, self).__init__()
                self.target_columns = target_columns
                self.history = {}
                for col in target_columns:
                    self.history[f'{col}_rmse'] = []
                    self.history[f'{col}_mae'] = []
                    self.history[f'{col}_r2'] = []
                    self.history[f'{col}_adj_r2'] = []
                    self.history[f'{col}_nse'] = []

            def on_epoch_end(self, epoch, logs=None):
                for col_idx, col in enumerate(self.target_columns):
                    y_true = y_test[:, col_idx]
                    y_pred = self.model.predict(X_meteorological_test.reshape(-1, X_meteorological_test.shape[1], 1))
                    if isinstance(y_pred, list):
                        y_pred = y_pred[col_idx].flatten()
                    else:
                        y_pred = y_pred.flatten()
                    
                    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae_value = mean_absolute_error(y_true, y_pred)
                    r2_value = r2_score(y_true, y_pred)
                    n = len(y_true)
                    p = X_meteorological_train.shape[1]
                    adj_r2_value = 1 - (1 - r2_value) * (n - 1) / (n - p - 1 - 1)  # Adjusted R2 formula
                    
                    mse_value = mean_squared_error(y_true, y_pred)
                    var_y = np.var(y_true)
                    nse_value = 1 - (mse_value / var_y)
                    
                    self.history[f'{col}_rmse'].append(rmse_value)
                    self.history[f'{col}_mae'].append(mae_value)
                    self.history[f'{col}_r2'].append(r2_value)
                    self.history[f'{col}_adj_r2'].append(adj_r2_value)
                    self.history[f'{col}_nse'].append(nse_value)

            def get_config(self):
                config = super().get_config()
                config.update({'target_columns': self.target_columns})
                return config

        start_time = datetime.now()

        input_shape = X_meteorological_train.shape[1], 1


        model = build_model(input_shape=input_shape, num_outputs=len(target_columns), **param_dict )

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # 创建自定义回调
        custom_callback = CustomCallback(target_columns)

        progress_callback = ProgressCallback(total_iterations, epochs, project_id)

        # 初始化历史记录
        total_history = {'loss': {}, 'val_loss': {}}
        if len(target_columns) == 1:
            total_history['loss']['loss'] = []
            total_history['val_loss']['val_loss'] = []
        else:
            for col in target_columns:
                total_history['loss'][f'output_{target_columns.index(col)}_loss'] = []
                total_history['val_loss'][f'val_output_{target_columns.index(col)}_loss'] = []

        # 训练模型
        if len(target_columns) > 1:
            history = model.fit(
                X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                {f'output_{i}': y_train[:, i] for i in range(len(target_columns))},
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping, custom_callback, progress_callback]
            )
        else:
            history = model.fit(
                X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping, custom_callback, progress_callback]
            )

        # 合并历史记录
        if len(target_columns) == 1:
            total_history['loss']['loss'].extend(history.history['loss'])
            total_history['val_loss']['val_loss'].extend(history.history['val_loss'])
        else:
            for col in target_columns:
                total_history['loss'][f'output_{target_columns.index(col)}_loss'].extend(
                    history.history[f'output_{target_columns.index(col)}_loss']
                )
                total_history['val_loss'][f'val_output_{target_columns.index(col)}_loss'].extend(
                    history.history[f'val_output_{target_columns.index(col)}_loss']
                )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # 收集评估指标
        eval_metrics = {}
        y_preds = model.predict(X_meteorological_test.reshape(-1, X_meteorological_test.shape[1], 1))
        for col_idx, col in enumerate(target_columns):
            y_true = y_test[:, col_idx]
            if isinstance(y_preds, list):
                y_pred = y_preds[col_idx].flatten()
            else:
                y_pred = y_preds.flatten()
            
            rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_value = mean_absolute_error(y_true, y_pred)
            r2_value = r2_score(y_true, y_pred)
            n = len(y_true)
            p = X_meteorological_train.shape[1]
            adj_r2_value = 1 - (1 - r2_value) * (n - 1) / (n - p - 1 - 1)  # Adjusted R2 formula
            
            mse_value = mean_squared_error(y_true, y_pred)
            var_y = np.var(y_true)
            nse_value = 1 - (mse_value / var_y)
            
            eval_metrics[col] = {
                'rmse': rmse_value,
                'mae': mae_value,
                'r2': r2_value,
                'adj_r2': adj_r2_value,
                'nse': nse_value
            }
        
        # 存储结果
        result = {
            'params': param_dict,
            'epochs': epochs,
            'batch_size': batch_size,
            'training_time': end_time - start_time,
            'history': total_history,
            **{metric: [eval_metrics[col][metric] for col in target_columns] for metric in ['rmse', 'mae', 'r2', 'adj_r2', 'nse']}
        }
        results.append(result)
    
        # 打印当前结果
        print(f"result: {result}")

        model_path = os.path.join('media', str(project_id), model_filename + '.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        num_outputs = len(target_columns)
        plot_loss(results, project_id, model_filename, num_outputs, target_columns)
        plot_metrics(results, project_id, model_filename, num_outputs, target_columns)
        plot_predictions(y_test, y_preds, project_id, model_filename, num_outputs, target_columns)

        if X_meteorological.shape[1] > 0:
            plot_feature_importance(scaler, X_meteorological, X_meteorological_scaled, project_id, model_filename, meteorological_features)

        has_meteorological_features = X_meteorological.shape[1] > 0
        combine_plots_to_pdf(project_id, model_filename, num_outputs, has_meteorological_features, X_meteorological, X_meteorological_scaled, scaler, meteorological_features)

        return {
            'status': 'success',
            'message': '模型训练并保存成功！',
            'params': param_dict
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        import traceback
        traceback.print_exc()


def train_multiple_param_combinations(data, target_columns, standard_date_features, meteorological_features, param_grid, project_id, test_size_ratio=0.2, weights=None):
    try:
       
        csv_file_path = os.path.join('media', str(project_id), 'data.csv')
        data.to_csv(csv_file_path, index=False)
        data = pd.read_csv(csv_file_path)

        if standard_date_features:
            date_column = standard_date_features[0]
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            data['year'] = data[date_column].dt.year
            data['month'] = data[date_column].dt.month
            data['day'] = data[date_column].dt.day
            meteorological_features.extend(['year', 'month', 'day'])

        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)  

        if X_meteorological.shape[1] > 0:
            scaler = StandardScaler()
            X_meteorological_scaled = scaler.fit_transform(X_meteorological)
        else:
            X_meteorological_scaled = X_meteorological

        # 数据准备
        X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
           X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)


        results = []
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        total_iterations = len(param_combinations)
        current_iteration = 0

        channel_layer = get_channel_layer()

        
        class ProgressCallback(Callback):
            def __init__(self, total_iterations, epochs, project_id):
                super().__init__()
                self.total_iterations = total_iterations
                self.epochs = epochs
                self.project_id = project_id
                self.current_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                self.current_epoch += 1
                if self.current_epoch == self.epochs:
                    nonlocal current_iteration
                    current_iteration += 1
                    progress = int((current_iteration / self.total_iterations) * 100)
                    print(f"Iteration {current_iteration}/{total_iterations}, Progress: {progress}%")
                    try:
                        async_to_sync(channel_layer.group_send)(
                            f'training_progress_{self.project_id}',
                            {
                                'type': 'send_progress',
                                'progress': progress
                            }
                        )
                    except Exception as e:
                        print(f"Error sending progress: {str(e)}")

        
        

        input_shape = X_meteorological_train.shape[1], 1

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            epochs = param_dict.pop('epochs')
            batch_size = param_dict.pop('batch_size')

            print('param_dict', param_dict)
            start_time = datetime.now()

            model = build_model(input_shape=input_shape, num_outputs=len(target_columns), **param_dict )

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            

            progress_callback = ProgressCallback(total_iterations, epochs, project_id)

    
            # 训练模型
            if len(target_columns) > 1:
                history = model.fit(
                    X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                    {f'output_{i}': y_train[:, i] for i in range(len(target_columns))},
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stopping,  progress_callback]
                )
            else:
                history = model.fit(
                    X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stopping,  progress_callback]
                )

           
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            # 收集评估指标
            eval_metrics = {}
            y_preds = model.predict(X_meteorological_test.reshape(-1, X_meteorological_test.shape[1], 1))
            for col_idx, col in enumerate(target_columns):
                y_true = y_test[:, col_idx]
                if isinstance(y_preds, list):
                    y_pred = y_preds[col_idx].flatten()
                else:
                    y_pred = y_preds.flatten()
                
                rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
                mae_value = mean_absolute_error(y_true, y_pred)
                r2_value = r2_score(y_true, y_pred)
                n = len(y_true)
                p = X_meteorological_train.shape[1]
                adj_r2_value = 1 - (1 - r2_value) * (n - 1) / (n - p - 1 - 1)  # Adjusted R2 formula
                
                mse_value = mean_squared_error(y_true, y_pred)
                var_y = np.var(y_true)
                nse_value = 1 - (mse_value / var_y)
                
                eval_metrics[col] = {
                    'rmse': rmse_value,
                    'mae': mae_value,
                    'r2': r2_value,
                    'adj_r2': adj_r2_value,
                    'nse': nse_value
                }
            
            # 计算 AIC 和 BIC
            num_samples = y_test.shape[0]
            num_params = model.count_params()
            mse = np.mean((y_pred - y_test) ** 2)
            aic = num_samples * np.log(mse) + 2 * num_params
            bic = num_samples * np.log(mse) + num_params * np.log(num_samples)
            
            # 存储结果
            result = {
                'params': param_dict,
                'epochs': epochs,
                'batch_size': batch_size,
                **{metric: [eval_metrics[col][metric] for col in target_columns] for metric in ['rmse', 'mae', 'r2', 'adj_r2', 'nse']},
                'training_time': training_time,
                'aic': aic,
                'bic': bic
            }
            results.append(result)
    
        # 打印当前结果
        print(f"result: {result}")

        if weights:
            evaluate_models(results, weights, project_id, target_columns)

        return {
            'status': 'success',
            'message': '模型训练并保存成功！',
            'params': param_dict
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


# def evaluate_models(results, weights, project_id, target_columns):
#     try:
#         calculated_results = []
        
#         for result in results:
#             single_result = {}
            
#             # 拆解 params 字典
#             for key, value in result['params'].items():
#                 if isinstance(value, list):
#                     value = ','.join(map(str, value))
#                 single_result[key] = value
            
#             # 添加其他字段
#             single_result['epochs'] = result['epochs']
#             single_result['batch_size'] = result['batch_size']
#             single_result['training_time'] = result['training_time']
            
#             num_targets = len(result['rmse'])
            
#             for i, target in enumerate(target_columns):
#                 avg_rmse = result['rmse'][i]
#                 avg_mae = result['mae'][i]
#                 avg_nse = result['nse'][i]
                
#                 single_result[f'rmse_{target}'] = avg_rmse
#                 single_result[f'mae_{target}'] = avg_mae
#                 single_result[f'r2_{target}'] = result['r2'][i]
#                 single_result[f'adj_r2_{target}'] = result['adj_r2'][i]
#                 single_result[f'nse_{target}'] = avg_nse
                
#                 single_result[f'calculate_{target}'] = (
#                     weights['w1'] * avg_rmse +
#                     weights['w2'] * avg_mae +
#                     weights['w3'] * (1 - avg_nse) +
#                     weights['w4'] * result['training_time']
#                 )
            
#             calculated_results.append(single_result)
        
#         results_df = pd.DataFrame(calculated_results)
#         csv_file_path = os.path.join('media', str(project_id), f'calculate_result.csv')
#         results_df.to_csv(csv_file_path, index=False)

#         print("Evaluation results saved to calculate_result.csv")

#         return {
#             'status': 'success',
#             'message': '模型评估成功！',
#             'results': calculated_results
#         }

#     except Exception as e:
#         print(f"Error in evaluate_models: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return {
#             'status': 'error',
#             'message': str(e)
#         }


from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from tensorflow.keras.callbacks import EarlyStopping


def train_with_bayesian_optimization(data, target_columns, standard_date_features, other_features, param_grid, project_id, model_filename, test_size_ratio=0.2,stop_event=None, n_calls=10, weights={'w1': 0.25, 'w2': 0.25, 'w3': 0.5}):
    try:
        print("Starting train_with_bayesian_optimization...")
        print(f"Target columns: {target_columns}")
        print(f"Standard date features: {standard_date_features}")
        print(f"Other features: {other_features}")
        print(f"Param grid: {param_grid}")
        print(f"Test size ratio: {test_size_ratio}")
        print(f"n_calls: {n_calls}")
        print(f"Weights: {weights}")

        csv_file_path = os.path.join('media', str(project_id), 'data.csv')
        data.to_csv(csv_file_path, index=False)
        data = pd.read_csv(csv_file_path)

        if standard_date_features:
            date_column = standard_date_features[0]
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            data['year'] = data[date_column].dt.year
            data['month'] = data[date_column].dt.month
            data['day'] = data[date_column].dt.day
            other_features.extend(['year', 'month', 'day'])

        X_meteorological = data[other_features].values.astype(float) if other_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)

        if X_meteorological.shape[1] > 0:
            scaler = StandardScaler()
            X_meteorological_scaled = scaler.fit_transform(X_meteorological)
        else:
            X_meteorological_scaled = X_meteorological

        # 数据准备
        X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
            X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)


        # 设置参数范围
        data_size = len(data)
        if data_size < 500:
            epochs_range = (50, 200)
            batch_size_range = (8, 64)
        elif data_size < 1000:
            epochs_range = (30, 100)
            batch_size_range = (16, 128)    
        elif data_size < 3000:
            epochs_range = (20, 80)
            batch_size_range = (16, 128)
        elif data_size < 5000:
            epochs_range = (10, 75)
            batch_size_range = (16, 128)
        else:
            epochs_range = (5, 50)
            batch_size_range = (32, 256)

        param_space = [
            Integer(epochs_range[0], epochs_range[1], name='epochs'),
            Integer(batch_size_range[0], batch_size_range[1], name='batch_size'),
            Integer(8, 128, name='filters'),
            Integer(2, 128, name='kernel_size'),
            Integer(16, 256, name='lstm_units'),
            Integer(16, 256, name='dense_units'),
            Integer(16, 256, name='dynamic_attention_units')
        ]

        # 定义目标函数
        @use_named_args(param_space)
        def objective(**params):
            epochs = params['epochs']
            batch_size = params['batch_size']
            filters = params['filters']
            kernel_size = (params['kernel_size'],)  # 将整数转换为元组
            lstm_units = params['lstm_units']
            dense_units = params['dense_units']
            dynamic_attention_units = params['dynamic_attention_units']

            input_shape = X_meteorological_train.shape[1], 1
            model = build_model(input_shape=input_shape, num_outputs=len(target_columns),
                                filters=filters, kernel_size=kernel_size, lstm_units=lstm_units,
                                dense_units=dense_units, dynamic_attention_units=dynamic_attention_units)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(
                X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping]
            )

            y_pred = model.predict(X_meteorological_test.reshape(-1, X_meteorological_test.shape[1], 1))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            nse = 1 - (mean_squared_error(y_test, y_pred) / np.var(y_test))
            score = weights['w1'] * rmse + weights['w2'] * mae + weights['w3'] * (1 - nse)*50

            return score

        # 使用贝叶斯优化
        total_iterations = n_calls
        current_iteration = 0

        channel_layer = get_channel_layer()

        class ProgressCallback_b(Callback):
            def __init__(self, total_iterations, project_id, channel_layer):
                super().__init__()
                self.total_iterations = total_iterations
                self.project_id = project_id
                self.channel_layer = channel_layer

            def update_progress(self):
                nonlocal current_iteration
                current_iteration += 1
                progress = int((current_iteration / self.total_iterations) * 100)
                print(f"Iteration {current_iteration}/{total_iterations}, Progress: {progress}%")
                try:
                    async_to_sync(self.channel_layer.group_send)(
                        f'training_progress_{self.project_id}',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'status': '正在查找最优参数组合（小范围参数组合训练）'
                        }
                    )
                except Exception as e:
                    print(f"Error sending progress: {str(e)}")

        # 使用闭包来处理传递的参数
        def create_callback(progress_callback):
            def callback(result):
                progress_callback.update_progress()
            return callback

        progress_callback = ProgressCallback_b(total_iterations, project_id, channel_layer)

        res_gp = gp_minimize(objective, param_space, n_calls=n_calls, random_state=0, callback=create_callback(progress_callback))

        # 重置进度条
        current_iteration = -1
        progress_callback.update_progress()

        class ProgressCallback(Callback):
            def __init__(self, total_iterations, epochs, project_id):
                super().__init__()
                self.total_iterations = total_iterations
                self.epochs = epochs
                self.project_id = project_id
                self.channel_layer = channel_layer
            def update_progress(self):
                nonlocal current_iteration
                current_iteration += 1

                progress = int((current_iteration / self.total_iterations) * 100)
                print(f"Iteration {current_iteration}/{self.total_iterations}, Progress: {progress}%")
                try:
                    async_to_sync(channel_layer.group_send)(
                        f'training_progress_{self.project_id}',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'status': '正在查找最优参数组合（小范围参数组合训练）'
                        }
                    )
                except Exception as e:
                    print(f"Error sending progress: {str(e)}")

        # 使用贝叶斯优化找到的最优参数组合构建小范围的参数组合
        best_params = res_gp.x
        best_params = [int(item) for item in best_params]
        print("Best Parameters:", best_params)

        small_param_grid = {
            'epochs': [best_params[0]],
            'batch_size': [best_params[1]],
            'filters': [best_params[2]],
            'kernel_size': [best_params[3]],
            'lstm_units': [best_params[4]],
            'dense_units': [best_params[5], best_params[3] + 1],
            'dynamic_attention_units': [best_params[6], best_params[6] + 1]
        }

        results = []
        param_names = list(small_param_grid.keys())
        param_values = [small_param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        best_result = None
        best_score = float('inf')
        total_iterations = len(param_combinations)

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            print('param_dict', param_dict)
            epochs = param_dict.pop('epochs')
            batch_size = param_dict.pop('batch_size')

            start_time = datetime.now()

            input_shape = X_meteorological_train.shape[1], 1
            model = build_model(input_shape=input_shape, num_outputs=len(target_columns), **param_dict)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            progress_callback = ProgressCallback(total_iterations, epochs, project_id)  


            history = model.fit(
                X_meteorological_train.reshape(-1, X_meteorological_train.shape[1], 1),
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stopping,progress_callback]
            )

            # 在训练结束后调用 update_progress 方法
            progress_callback.update_progress()

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            # 检查是否触发了 EarlyStopping
            # if early_stopping.stopped_epoch == 0:
            y_pred = model.predict(X_meteorological_test.reshape(-1, X_meteorological_test.shape[1], 1))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            nse = 1 - (mean_squared_error(y_test, y_pred) / np.var(y_test))
            score = weights['w1'] * rmse + weights['w2'] * mae + weights['w3'] * (1 - nse)*50

            results.append({
                'params': param_dict,
                'epochs': epochs,
                'batch_size': batch_size,
                'rmse': rmse,
                'mae': mae,
                'r2': r2_score(y_test, y_pred),
                'adj_r2': 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_meteorological_train.shape[1] - 1 - 1),
                'nse': nse,
                'training_time': (datetime.now() - start_time).total_seconds(),
                'history': history.history
            })

            if score < best_score:
                best_score = score
                best_result = {
                    'params': param_dict,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2_score(y_test, y_pred),
                    'adj_r2': 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_meteorological_train.shape[1] - 1 - 1),
                    'nse': nse,
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'history': history.history,
                    'model': model,
                    'score': score
                }

        if not results:
            print("No parameter combinations completed all epochs without early stopping.")
            return {
                'status': 'error',
                'message': '所有参数组合均触发了 EarlyStopping，未找到合适的参数组合。'
            }

        if best_result:
            best_model_filename = 'best_model'
            best_model_path = os.path.join('media', str(project_id), f'{best_model_filename}.h5')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            best_result['model'].save(best_model_path)

            # 准备返回的数据
            best_params = best_result['params']
            epochs = best_result['epochs']
            batch_size = best_result['batch_size']
            rmse = best_result['rmse']
            mae = best_result['mae']
            r2 = best_result['r2']
            adj_r2 = best_result['adj_r2']
            nse = best_result['nse']
            training_time = best_result['training_time']
            score = best_result['score']

            best_result_data = {
            'best_params': best_params,
            'epochs': epochs,
            'batch_size': batch_size,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'nse': nse,
            'training_time': training_time,
            'score': score
            }
            print("best_result_data:", best_result_data)

            processed_best_result_data = process_best_result_data(best_result_data)

            # 将 best_result_data 写入 JSON 文件
            best_result_path = os.path.join('media', str(project_id), f'{model_filename}_best_result.json')
            with open(best_result_path, 'w') as f:
                json.dump(processed_best_result_data, f)


            return {
                'status': 'success',
                'message': '模型训练并保存成功！',
                'best_params': best_result['params'],
                'score': best_result['score']
            }

    except Exception as e:
        print(f"Error in train_with_bayesian_optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


def process_best_result_data(best_result_data):
    # 处理 best_params 中的 numpy.int64 类型
    for key, value in best_result_data['best_params'].items():
        if isinstance(value, np.int64):
            best_result_data['best_params'][key] = int(value)
        elif isinstance(value, tuple):
            best_result_data['best_params'][key] = tuple(int(v) if isinstance(v, np.int64) else v for v in value)
        elif isinstance(value, list):
            best_result_data['best_params'][key] = [int(v) if isinstance(v, np.int64) else v for v in value]

    # 处理 epochs 和 batch_size
    if isinstance(best_result_data['epochs'], np.int64):
        best_result_data['epochs'] = int(best_result_data['epochs'])
    if isinstance(best_result_data['batch_size'], np.int64):
        best_result_data['batch_size'] = int(best_result_data['batch_size'])

    # 处理 rmse, mae, r2, adj_r2, nse
    best_result_data['rmse'] = [float(best_result_data['rmse'])] if isinstance(best_result_data['rmse'], np.float64) else best_result_data['rmse']
    best_result_data['mae'] = [float(best_result_data['mae'])] if isinstance(best_result_data['mae'], np.float64) else best_result_data['mae']
    best_result_data['r2'] = [float(best_result_data['r2'])] if isinstance(best_result_data['r2'], np.float64) else best_result_data['r2']
    best_result_data['adj_r2'] = [float(best_result_data['adj_r2'])] if isinstance(best_result_data['adj_r2'], np.float64) else best_result_data['adj_r2']
    best_result_data['nse'] = [float(best_result_data['nse'])] if isinstance(best_result_data['nse'], np.float64) else best_result_data['nse']

    # 处理 training_time
    if isinstance(best_result_data['training_time'], np.float64):
        best_result_data['training_time'] = float(best_result_data['training_time'])

    return best_result_data


def predict_model(df, target_columns, standard_date_features, other_features, project_id, train_df):
    try:
        # 定义自定义层
        custom_objects = {
            'CustomAttention': CustomAttention
        }
        
        model_filename = f'model_{project_id}.h5'
        model_path = os.path.join('media', str(project_id), model_filename)
        # 加载模型
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        if standard_date_features:
            date_column = standard_date_features[0]

            train_df[date_column] = pd.to_datetime(train_df[date_column])
            train_df = train_df.sort_values(by=date_column)
            train_df['year'] = train_df[date_column].dt.year
            train_df['month'] = train_df[date_column].dt.month
            train_df['day'] = train_df[date_column].dt.day
            other_features.extend(['year', 'month', 'day'])

            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day

        X_train_other = train_df[other_features].values.astype(float)

        scaler = StandardScaler()
        scaler.fit(X_train_other)
        X_train_other_scaled = scaler.transform(X_train_other)

        X_other = df[other_features].values.astype(float)
        X_other_scaled = scaler.transform(X_other)

        # 准备数据
        X_other_scaled = X_other_scaled.reshape(X_other_scaled.shape[0], X_other_scaled.shape[1], 1)  # LSTM 需要 3D 输入

        print(f"X_other_scaled shape: {X_other_scaled.shape}")

        y_pred = model.predict(X_other_scaled)

        if len(target_columns) == 1:
            df[f'predict_{target_columns[0]}'] = y_pred.flatten()
        else:
            y_pred_list = model.predict(X_other_scaled)
            y_pred = np.column_stack(y_pred_list)
            for i, target in enumerate(target_columns):
                df[f'predict_{target}'] = y_pred[:, i]

        # Save the predicted data
        predicted_csv_file_path = os.path.join('media', str(project_id), f'predicted_result.csv')
        df.to_csv(predicted_csv_file_path, index=False)

        return {
            'status': 'success',
            'message': '预测完成并保存成功！',
            'predicted_file': f'predicted_result.csv'
        }

    except Exception as e:
        print(f"Error in traditional_predict_model: {str(e)}")
        import traceback
        traceback.print_exc()  
        return {
            'status': 'error',
            'message': str(e)
        }