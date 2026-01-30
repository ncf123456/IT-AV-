import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, Conv1D, LSTM, Add, Concatenate, Lambda, ZeroPadding1D, Cropping1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,Callback
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from io import BytesIO
import seaborn as sns
import base64
import time
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import joblib
from datetime import datetime
from fpdf import FPDF
from .models import DataFile, ModelFile
import json
matplotlib.use('Agg')

# 自定义层 TemporalAwareConv1D
class TemporalAwareConv1D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(TemporalAwareConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.dense_time = Dense(32, activation='relu')
        self.conv = Conv1D(self.filters, self.kernel_size, padding='same', activation='relu')
        self.time_weights = self.add_weight(name='time_weights',
                                            shape=(input_shape[1], 1),
                                            initializer='ones',
                                            trainable=True)

    def call(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Input data must have shape (batch_size, timesteps, features)")
        
        # 提取时间特征
        time_features = inputs[:, :, :3]
        
        # 处理时间特征
        time_features = self.dense_time(time_features)
        
        # 组合时间特征和其他特征
        combined_input = Concatenate(axis=-1)([time_features, inputs[:, :, 3:]])
        
        # 卷积操作
        conv_output = self.conv(combined_input)
        
        # 应用时间权重
        time_weights_broadcasted = tf.expand_dims(self.time_weights, axis=0)  # 添加 batch 维度
        weighted_output = conv_output * time_weights_broadcasted
        
        return weighted_output

    def get_config(self):
        config = super(TemporalAwareConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

class SpatialAwareConv1D(Layer):
    def __init__(self, filters=32, kernel_sizes=[3, 5], strides=1, padding='valid', **kwargs):
        super(SpatialAwareConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.convs = None
        self.dense_geo = None
        self.spatial_weights = None
        self.geo_weights = None

    def build(self, input_shape):
        if self.convs is None:
            self.convs = [Conv1D(self.filters, k, strides=self.strides, padding=self.padding, activation='relu') for k in self.kernel_sizes]
        
        if self.dense_geo is None:
            self.dense_geo = Dense(32, activation='relu')
        
        if self.spatial_weights is None:
            self.spatial_weights = self.add_weight(name='spatial_weights',
                                                   shape=(len(self.kernel_sizes),),
                                                   initializer='ones',
                                                   trainable=True)
        
        if self.geo_weights is None:
            self.geo_weights = self.add_weight(name='geo_weights',
                                               shape=(1,),
                                               initializer='ones',
                                               trainable=True)

    def call(self, inputs):
        batch_size, timesteps, features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        # 提取地理特征
        geo_features = inputs[:, :, :2]
        data_features = inputs[:, :, 2:]
        
        # 处理地理特征
        geo_features = self.dense_geo(geo_features)
        
        # 组合地理特征和其他特征
        combined_input = Concatenate(axis=-1)([geo_features, data_features])
        
        # 计算需要的最大时间步长
        max_kernel_size = tf.reduce_max(self.kernel_sizes)
        required_timesteps = max_kernel_size
        
        # 如果时间步长不足，则进行零填充
        padding_needed = required_timesteps - timesteps
        padded_combined_input = tf.cond(
            padding_needed > 0,
            lambda: tf.pad(combined_input, [[0, 0], [0, padding_needed], [0, 0]], mode='CONSTANT', constant_values=0),
            lambda: combined_input
        )
        
        # 对组合后的输入应用多个卷积核
        conv_outputs = [conv(padded_combined_input) for conv in self.convs]
        
        # 确保所有卷积输出具有相同的时间步长
        time_steps = [tf.shape(output)[1] for output in conv_outputs]
        max_timesteps = tf.reduce_max(time_steps)
        
        # 确保所有卷积输出具有相同的时间步长
        padded_conv_outputs = [tf.pad(output, [[0, 0], [0, max_timesteps - tf.shape(output)[1]], [0, 0]]) for output in conv_outputs]
        
        # 合并卷积输出
        stacked_conv_outputs = tf.stack(padded_conv_outputs, axis=0)
        
        # 应用空间权重
        spatial_weights = tf.reshape(self.spatial_weights, (len(self.kernel_sizes), 1, 1, 1))
        weighted_conv_outputs = stacked_conv_outputs * spatial_weights
        
        # 汇总加权后的卷积输出
        combined_output = tf.reduce_sum(weighted_conv_outputs, axis=0)
        
        # 应用地理权重
        geo_weights = tf.expand_dims(self.geo_weights, axis=-1)
        combined_output = combined_output * geo_weights
        
        return combined_output

    def get_config(self):
        config = super(SpatialAwareConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

class TimeAwareAttention(Layer):
    def __init__(self, dynamic=False, **kwargs):
        super(TimeAwareAttention, self).__init__(dynamic=dynamic, **kwargs)

    def build(self, input_shape):
        # 确保输入形状是一个列表
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `TimeAwareAttention` layer should be called '
                             'on a list of 2 inputs.')
        
        self.time_dense = Dense(32, activation='relu')
        
        # 获取特征维度
        feature_dim = input_shape[0][-1] + 32
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(feature_dim, 1),
                                                 initializer='uniform',
                                                 trainable=True)

    def call(self, inputs):
        # 解包输入
        lstm_out, time_features = inputs
        
        # 处理时间特征
        time_features = self.time_dense(time_features)
        
        # 动态获取时间步长
        lstm_time_steps = tf.shape(lstm_out)[1]
        time_feature_time_steps = tf.shape(time_features)[1]

        # 如果 time_features 的时间步长与 lstm_out 不一致，则扩展或截断 time_features
        repeat_factor = lstm_time_steps // time_feature_time_steps
        tiled_time_features = tf.tile(time_features, [1, repeat_factor, 1])
        padding_size = tf.maximum(0, lstm_time_steps - (repeat_factor * time_feature_time_steps))
        padded_time_features = tf.pad(tiled_time_features, [[0, 0], [0, padding_size], [0, 0]])

        # 组合时间特征和其他特征
        combined_input = Concatenate(axis=-1)([lstm_out, padded_time_features])
        
        # 计算注意力分数
        attention_scores = tf.matmul(combined_input, self.attention_weights)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)

        # 加权求和
        context_vector = lstm_out * attention_scores
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        config = super(TimeAwareAttention, self).get_config()
        return config

# 位置编码
class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]  # 获取序列长度
        d_model = tf.shape(inputs)[2]  # 获取特征维度

        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))

        # 计算 sin 和 cos 位置编码
        pos_sin = tf.sin(position * div_term)
        pos_cos = tf.cos(position * div_term)

        # 合并 sin 和 cos 位置编码
        positional_encoding = tf.concat([pos_sin, pos_cos], axis=-1)

        # 如果 d_model 是奇数，添加一个全零列
        def true_fn():
            return tf.concat([positional_encoding, tf.zeros((seq_length, 1))], axis=-1)

        def false_fn():
            return positional_encoding

        positional_encoding = tf.cond(tf.equal(d_model % 2, 1), true_fn, false_fn)

        # 确保位置编码的最后一个维度与 d_model 一致
        positional_encoding = positional_encoding[:, :d_model]

        # 将位置编码扩展到与输入相同的形状
        positional_encoding = tf.expand_dims(positional_encoding, 0)  # 添加 batch 维度
        positional_encoding = tf.tile(positional_encoding, [tf.shape(inputs)[0], 1, 1])  # 复制 batch 次

        return inputs + positional_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        return config

# 动态调整注意力权重
class DynamicAttention(Layer):
    def __init__(self, units, **kwargs):
        super(DynamicAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.gru = LSTM(self.units, return_sequences=True)
        self.dense = Dense(1)

    def call(self, inputs):
        # 使用 GRU 动态调整注意力权重
        gru_output = self.gru(inputs)
        attention_scores = self.dense(gru_output)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)

        # 加权求和
        context_vector = inputs * attention_scores
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        config = super(DynamicAttention, self).get_config()
        config.update({
            'units': self.units
        })
        return config

# 加权融合
class WeightedFusion(Layer):
    def __init__(self, **kwargs):
        super(WeightedFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        # 确保 fusion_weights 的形状与输入的数量匹配
        self.fusion_weights = self.add_weight(name='fusion_weights',
                                              shape=(len(input_shape), 1),
                                              initializer='uniform',
                                              trainable=True)

    def call(self, inputs):
        # 计算融合权重
        fusion_weights = tf.nn.softmax(self.fusion_weights, axis=0)
        
        # 将 fusion_weights 扩展到与输入相同的维度
        fusion_weights = tf.expand_dims(fusion_weights, axis=-1)
        
        # 加权融合
        fused_output = tf.reduce_sum(inputs * fusion_weights, axis=0)
        
        return fused_output

    def get_config(self):
        return super(WeightedFusion, self).get_config()
    
# 自定义 R² 指标
def r_square(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# 构建模型
def build_temporal_aware_lstm(input_shape, time_steps, geographical_steps, target_columns, lstm_units, dense_units=32, filters=64, temporal_kernel_size=3, spatial_kernel_sizes=[3, 5], dynamic_attention_units=32):
    input_layer = Input(shape=input_shape, name='input')
    
    # 使用 Lambda 层拆分时间特征、地理位置特征和气象特征
    time_features = Lambda(lambda x: x[:, :time_steps, :], name='time_features')(input_layer)
    geographical_features = Lambda(lambda x: x[:, time_steps:time_steps+geographical_steps, :], name='geographical_features')(input_layer)
    meteorological_features = Lambda(lambda x: x[:, time_steps+geographical_steps:, :], name='meteorological_features')(input_layer)

    # 计算所有特征中的最大时间步数
    max_time_steps = max(time_steps, geographical_steps, meteorological_features.shape[1])

    # 调整时间特征的形状
    if time_steps < max_time_steps:
        time_features_padded = ZeroPadding1D(padding=(0, max_time_steps - time_steps), name='time_features_padded')(time_features)
    else:
        time_features_padded = Cropping1D(cropping=(0, time_steps - max_time_steps), name='time_features_cropped')(time_features)

    # 调整地理特征的形状
    if geographical_steps < max_time_steps:
        geographical_features_padded = ZeroPadding1D(padding=(0, max_time_steps - geographical_steps), name='geographical_features_padded')(geographical_features)
    else:
        geographical_features_padded = Cropping1D(cropping=(0, geographical_steps - max_time_steps), name='geographical_features_cropped')(geographical_features)

    # 调整气象特征的形状
    if meteorological_features.shape[1] < max_time_steps:
        meteorological_features_padded = ZeroPadding1D(padding=(0, max_time_steps - meteorological_features.shape[1]), name='meteorological_features_padded')(meteorological_features)
    else:
        meteorological_features_padded = Cropping1D(cropping=(0, meteorological_features.shape[1] - max_time_steps), name='meteorological_features_cropped')(meteorological_features)

    # 组合时间特征和气象特征
    combined_time_meteorological = Concatenate(axis=-1, name='combined_time_meteorological')([time_features_padded, meteorological_features_padded])

    # 组合地理特征和气象特征
    combined_geographical_meteorological = Concatenate(axis=-1, name='combined_geographical_meteorological')([geographical_features_padded, meteorological_features_padded])

    # 添加 TemporalAwareConv1D 层
    temporal_conv_out = TemporalAwareConv1D(filters=filters, kernel_size=temporal_kernel_size, name='temporal_conv')(combined_time_meteorological)

    # 添加 SpatialAwareConv1D 层来处理地理特征
    spatial_conv_out = SpatialAwareConv1D(filters=filters, kernel_sizes=spatial_kernel_sizes, name='spatial_conv')(combined_geographical_meteorological)

    # 调整 meteorological_features 的形状以匹配 temporal_conv_out 和 spatial_conv_out
    adjusted_meteorological_features = Conv1D(filters=filters, kernel_size=1, padding='same', name='adjust_meteorological')(meteorological_features_padded)

    # 合并时间特征、地理位置特征和气象特征
    combined_input = Concatenate(axis=1, name='concatenate')([temporal_conv_out, spatial_conv_out, adjusted_meteorological_features])

    # 位置编码
    positional_encoded = PositionalEncoding(name='positional_encoding')(combined_input)

    # LSTM 层
    lstm_out = LSTM(lstm_units, return_sequences=True, name='lstm')(positional_encoded)

    # 时间感知注意力
    time_aware_attention = TimeAwareAttention(name='time_aware_attention')([lstm_out, time_features_padded])

    # 动态调整注意力权重
    dynamic_attention = DynamicAttention(units=dynamic_attention_units, name='dynamic_attention')(lstm_out)

    # 加权融合
    weighted_fusion = WeightedFusion(name='weighted_fusion')([time_aware_attention, dynamic_attention])

    # 全连接层
    dense_out = Dense(dense_units, activation='relu', name='dense')(weighted_fusion)

    # 输出层
    output = Dense(len(target_columns), name='output')(dense_out)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae', r_square])
    
    if len(target_columns) > 1:
        outputs = [Dense(1, name=f'output_{i}')(dense_out) for i in range(len(target_columns))]
        model_loss = Model(inputs=input_layer, outputs=outputs)
        
        loss_functions = {f'output_{i}': 'mean_squared_error' for i in range(len(target_columns))}
        model_loss.compile(optimizer=Adam(learning_rate=0.001), loss=loss_functions, metrics=['mae', r_square])
        return model, model_loss
    else:
        return model, None

def plot_loss(results, project_id, model_filename, num_outputs, target_columns):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    
    for i, result in enumerate(results):
        history = result['history']
        for j in range(num_outputs):
            train_key = f'output_{j}_loss' if num_outputs > 1 else 'loss'
            val_key = f'val_output_{j}_loss' if num_outputs > 1 else 'val_loss'
            
            if train_key not in history:
                print(f"Warning: {train_key} not found in history. Using 'loss' instead.")
                train_key = 'loss'
            
            if val_key not in history:
                print(f"Warning: {val_key} not found in history. Using 'val_loss' instead.")
                val_key = 'val_loss'
            
            plt.plot(history[train_key], label=f'Train Loss ({target_columns[j]})', color=colors[j*2 % len(colors)])
            plt.plot(history[val_key], label=f'Val Loss ({target_columns[j]})', linestyle='--', color=colors[j*2+1 % len(colors)])
    
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
        axs[j].plot(y_true[:, j], label='True Values')
        axs[j].plot(y_pred[:, j], label='Predicted Values')
        axs[j].set_xlabel('Samples')
        axs[j].set_ylabel('Values')
        axs[j].legend()
        axs[j].set_title(f'True vs Predicted Values for {target_columns[j]}')

    plot_path = os.path.join('media', str(project_id), f'{model_filename}_predictions.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def plot_feature_importance(scaler_time, scaler_geographical, scaler_meteorological,
                             X_time, X_geographical, X_meteorological,
                             X_time_scaled, X_geographical_scaled, X_meteorological_scaled,
                             project_id, model_filename, time_features, geographical_features, meteorological_features):
    # Combine all features and their scaled means
    all_features = np.concatenate([X_time, X_geographical, X_meteorological], axis=1)
    all_features_scaled = np.concatenate([X_time_scaled, X_geographical_scaled, X_meteorological_scaled], axis=1)
    
    feature_names = time_features + geographical_features + meteorological_features
    
    original_feature_names = feature_names  # Use provided feature names

    scaled_feature_means = all_features_scaled.mean(axis=0)
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


class CustomEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.stopped_epoch = 0
        self.wait = 0
        self.best = np.Inf
        self.triggered_at_epoch = -1  # 新增属性，记录早停触发的轮次

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: Early stopping requires {self.monitor} available!")
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and self.triggered_at_epoch == -1:
                self.stopped_epoch = epoch
                self.triggered_at_epoch = epoch + 1  # 记录早停触发的轮次
                print(f"Early stopping would have triggered at epoch {epoch + 1}.")


import asyncio
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
def train_model(data, target_columns, standard_date_features, time_features, geographical_features, meteorological_features, param_grid, project_id, model_filename, test_size_ratio=0.2):
    try:
        print("Starting train_model...")
        print(f"Target columns: {target_columns}")
        print(f"Standard date features: {standard_date_features}")
        print(f"Time features: {time_features}")
        print(f"Geographic features: {geographical_features}")
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
            time_features.extend(['year', 'month', 'day'])

        X_time = data[time_features].values.astype(float)
        X_geographical = data[geographical_features].values.astype(float)
        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)  

        print(f"X_time shape: {X_time.shape}")
        print(f"X_geographical shape: {X_geographical.shape}")
        print(f"X_meteorological shape: {X_meteorological.shape}")
        print(f"y shape: {y.shape}")

        scalers = {}
        for feature_type, X in zip(['time', 'geographical', 'meteorological'], [X_time, X_geographical, X_meteorological]):
            if X.size > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scalers[f'X_{feature_type}'] = (scaler, X_scaled)
            else:
                scalers[f'X_{feature_type}'] = (None, X)

        X_time_scaler, X_time_scaled = scalers['X_time']
        X_geographical_scaler, X_geographical_scaled = scalers['X_geographical']
        X_meteorological_scaler, X_meteorological_scaled = scalers['X_meteorological']

        

        # 数据准备
        X_time_train, X_time_test, X_geographical_train, X_geographical_test, X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
            X_time_scaled, X_geographical_scaled, X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)

        # 准备数据
        X_time_train = X_time_train.reshape(X_time_train.shape[0], X_time_train.shape[1], 1)  # LSTM 需要 3D 输入
        X_time_test = X_time_test.reshape(X_time_test.shape[0], X_time_test.shape[1], 1)
        X_geographical_train = X_geographical_train.reshape(X_geographical_train.shape[0], X_geographical_train.shape[1], 1)
        X_geographical_test = X_geographical_test.reshape(X_geographical_test.shape[0], X_geographical_test.shape[1], 1)
        X_meteorological_train = X_meteorological_train.reshape(X_meteorological_train.shape[0], X_meteorological_train.shape[1], 1)
        X_meteorological_test = X_meteorological_test.reshape(X_meteorological_test.shape[0], X_meteorological_test.shape[1], 1)

        # 合并输入数据
        X_train = np.concatenate([X_time_train, X_geographical_train, X_meteorological_train], axis=1)
        X_test = np.concatenate([X_time_test, X_geographical_test, X_meteorological_test], axis=1)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

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

        start_time = datetime.now()

        input_shape = X_train.shape[1:]
        time_steps = len(time_features)
        geographical_steps = len(geographical_features)

        model, model_loss = build_temporal_aware_lstm(
            input_shape=input_shape,
            time_steps=time_steps,
            geographical_steps=geographical_steps,
            target_columns=target_columns,
            **param_dict  
        )

        custom_early_stopping = CustomEarlyStopping(monitor='val_loss', patience=10)

        progress_callback = ProgressCallback(total_iterations, epochs, project_id)

        # 获取训练和验证损失
        train_loss = {}
        val_loss = {}

        if model_loss is not None:
            history1 = model_loss.fit(
                X_train,
                [y_train[:, i] for i in range(len(target_columns))],
                validation_data=(X_test, [y_test[:, i] for i in range(len(target_columns))]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[custom_early_stopping, progress_callback],
                verbose=0
            )
            triggered_epoch = custom_early_stopping.triggered_at_epoch  # 获取早停触发的轮次
        
            y_pred_list = model_loss.predict(X_test)
            y_pred = np.column_stack(y_pred_list)

            # 获取训练和验证损失
            train_loss = history1.history['loss']
            val_loss = history1.history['val_loss']
        else:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[custom_early_stopping, progress_callback],
                verbose=0
            )
            triggered_epoch = custom_early_stopping.triggered_at_epoch  # 获取早停触发的轮次
            y_pred = model.predict(X_test)

            # 获取训练和验证损失
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']


        

        print("triggered_epoch:",triggered_epoch)
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))
        mae = np.mean(np.abs(y_pred - y_test), axis=0)
        nse = 1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))
        r2 = 1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(meteorological_features) - 1 - 1)
        
        if model_loss is not None:
            results.append({
                'params': param_dict,
                'epochs': epochs,
                'batch_size': batch_size,
                'rmse': rmse.tolist(),
                'mae': mae.tolist(),
                'r2': r2.tolist(),
                'adj_r2': adj_r2.tolist(),
                'nse': nse.tolist(),
                'training_time': training_time,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history1.history,
            })
        else:
            results.append({
                'params': param_dict,
                'epochs': epochs,
                'batch_size': batch_size,
                'rmse': rmse.tolist(),
                'mae': mae.tolist(),
                'r2': r2.tolist(),
                'adj_r2': adj_r2.tolist(),
                'nse': nse.tolist(),
                'training_time': training_time,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history.history,
            })

        # print(results)

        model_path = os.path.join('media', str(project_id), model_filename + '.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if model_loss is not None:
            model_loss.save(model_path)
        else:
            model.save(model_path)

        num_outputs = len(target_columns)
        plot_loss(results, project_id, model_filename, num_outputs, target_columns)
        plot_metrics(results, project_id, model_filename, num_outputs, target_columns)
        plot_predictions(y_test, y_pred, project_id, model_filename, num_outputs, target_columns)

        if X_meteorological.shape[1] > 0:
            plot_feature_importance(
                X_time_scaler, X_geographical_scaler, X_meteorological_scaler,
                X_time, X_geographical, X_meteorological,
                X_time_scaled, X_geographical_scaled, X_meteorological_scaled,
                project_id, model_filename, time_features, geographical_features, meteorological_features
            )

        has_meteorological_features = X_meteorological.shape[1] > 0
        combine_plots_to_pdf(project_id, model_filename, num_outputs, has_meteorological_features, X_meteorological, X_meteorological_scaled, scaler, meteorological_features)

        # 调用参数建议方法
        analyze_results_and_suggest_params(results, project_id, model_filename, num_outputs, target_columns,triggered_epoch)

        return {
            'status': 'success',
            'message': '模型训练并保存成功！',
            'results': results,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'scalers': scalers,
            'X_time': X_time.tolist(),
            'X_geographical': X_geographical.tolist(),
            'X_meteorological': X_meteorological.tolist(),
            'X_time_scaled': X_time_scaled.tolist(),
            'X_geographical_scaled': X_geographical_scaled.tolist(),
            'X_meteorological_scaled': X_meteorological_scaled.tolist(),
            'triggered_epoch': triggered_epoch
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        import traceback
        traceback.print_exc()  
        return {
            'status': 'error',
            'message': str(e)}
    

def analyze_results_and_suggest_params(results, project_id, model_filename, num_outputs, target_columns, triggered_epoch=None):
    try:
        # Load previous results if they exist
        previous_results_path = os.path.join('media', str(project_id), f'{model_filename}_metrics.json')
        if os.path.exists(previous_results_path):
            with open(previous_results_path, 'r') as f:
                previous_results = json.load(f)
        else:
            previous_results = []

        # Load previous params if they exist
        previous_params_path = os.path.join('media', str(project_id), f'{model_filename}_params.json')
        if os.path.exists(previous_params_path):
            with open(previous_params_path, 'r') as f:
                previous_params = json.load(f)
        else:
            previous_params = None

        suggestions = []

        for i, result in enumerate(results):
            current_rmse = result['rmse']
            current_mae = result['mae']
            current_r2 = result['r2']
            current_adj_r2 = result['adj_r2']
            current_nse = result['nse']
            current_training_time = result['training_time']
            current_params = result['params']

            # Save current params
            with open(previous_params_path, 'w', encoding='utf-8') as f:
                json.dump(current_params, f, ensure_ascii=False, indent=4)

            # Compare with previous results if available
            if previous_results:
                previous_rmse = previous_results[i]['rmse']
                previous_mae = previous_results[i]['mae']
                previous_r2 = previous_results[i]['r2']
                previous_adj_r2 = previous_results[i]['adj_r2']
                previous_nse = previous_results[i]['nse']
                previous_training_time = previous_results[i]['time']

            param_suggestions = {}

            
            

            for j, target in enumerate(target_columns):
                time_increase = (current_training_time - previous_training_time[j]) / previous_training_time[j]

                # 分析训练和验证损失
                train_loss = result['train_loss']
                val_loss = result['val_loss']

                # 定义容忍度阈值
                loss_tolerance = 1.5

                if current_rmse[j] > 3.0 or current_mae[j] > 2.5:
                    param_suggestions['lstm_units'] = '大幅变化'
                    param_suggestions['lstm_units_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑大幅变化 LSTM 单元数，会根据下一次的结果推荐变化趋势'
                    param_suggestions['dense_units'] = '大幅变化，一般为增大'
                    param_suggestions['dense_units_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑大幅变化 Dense 单元数，会根据下一次的结果推荐变化趋势'
                else:
                    param_suggestions['lstm_units'] = '小幅变化或不变'
                    param_suggestions['lstm_units_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 相对良好，考虑稍微变化或不变 LSTM 单元数，会根据下一次的结果推荐变化趋势'
                    param_suggestions['dense_units'] = '小幅变化（一般为增大）或不变'
                    param_suggestions['dense_units_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 相对良好，考虑稍微变化或不变 Dense 单元数，会根据下一次的结果推荐变化趋势'

                if triggered_epoch is not None and triggered_epoch>1:
                    param_suggestions['epochs'] = '稍稍变小'
                    param_suggestions['epochs_reason'] = f'早停触发于第 {triggered_epoch} 轮，建议将 epochs 减少到小于{triggered_epoch}的数值'

                # NSE
                if current_nse[j] < 0.9:
                    param_suggestions['spatial_kernel_sizes'] = '大幅改变'
                    param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.9，考虑增加卷积核大小，会根据下一次的结果推荐变化趋势'
                    param_suggestions['dynamic_attention_units'] = '大幅改变，一般为变大'
                    param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.9，考虑增加动态注意力单位数，会根据下一次的结果推荐变化趋势'
                    if triggered_epoch is None or triggered_epoch<=1:
                        param_suggestions['epochs'] = '变大'
                        param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.9，考虑增加训练轮数，会根据下一次的结果推荐变化趋势'
                
                else:
                    param_suggestions['spatial_kernel_sizes'] = '小幅变化或不变'
                    param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑改变或不变卷积核大小，会根据下一次的结果推荐变化趋势'
                    param_suggestions['dynamic_attention_units'] = '小幅变化（一般为增大）或不变'
                    param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑改变或不变动态注意力单位数，会根据下一次的结果推荐变化趋势'
                    if triggered_epoch is None or triggered_epoch<=1:
                        param_suggestions['epochs'] = '稍稍变大或不变'
                        param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变训练轮数，会根据下一次的结果推荐变化趋势'

                # Filters and Temporal Kernel Size
                if current_rmse[j] > 3.0 or current_mae[j] > 2.5:
                    param_suggestions['filters'] = '大幅变化'
                    param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑大幅变化滤波器数量，会根据下一次的结果推荐变化趋势'
                    param_suggestions['temporal_kernel_size'] = '大幅变化'
                    param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑大幅变化时间卷积核大小，会根据下一次的结果推荐变化趋势'
                
                else:
                    param_suggestions['filters'] = '小幅变化或不变'
                    param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f}) 相对良好，考虑稍微变化或不变滤波器数量，会根据下一次的结果推荐变化趋势'
                    param_suggestions['temporal_kernel_size'] = '小幅变化或不变'
                    param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f}) 相对良好，考虑稍微变化或不变时间卷积核大小'

                # Batch Size
                if train_loss[-1] > val_loss[-1] and val_loss[-1] < 3:
                    param_suggestions['batch_size'] = '稍稍变大'
                    param_suggestions['batch_size_reason'] = f'训练过程稳定且验证损失较低 ({val_loss[-1]:.4f})，考虑增大 batch_size'
                elif train_loss[-1] < 3 and val_loss[-1] < 3:
                    param_suggestions['batch_size'] = '稍稍变大'
                    param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 均较低且稳定，考虑增大 batch_size'
                elif train_loss[-1] > val_loss[-1] and val_loss[-1] > 3:
                    param_suggestions['batch_size'] = '稍稍变小'
                    param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 高于验证损失 ({val_loss[-1]:.4f}) 且验证损失较高，考虑减小 batch_size'
                elif train_loss[-1] > 3 and val_loss[-1] > 3:
                    param_suggestions['batch_size'] = '稍稍变小'
                    param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 较高，考虑减小 batch_size'
                else:
                    param_suggestions['batch_size'] = '保持不变'
                    param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 差异不大，考虑保持 batch_size 不变'

                if previous_results and previous_params:
                    # Adjust suggestions based on param changes
                    param_changes = {k: (current_params[k], previous_params[k]) for k in current_params if current_params[k] != previous_params[k]}
                    param_changes_reasons = {}

                    for param, (current_val, previous_val) in param_changes.items():
                        if param in param_suggestions:
                            if param in ['lstm_units', 'dense_units']:
                                if current_val > previous_val:
                                    if current_rmse[j] < previous_rmse[j] and current_mae[j] < previous_mae[j]:
                                        param_suggestions[param] = '继续变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，且 RMSE 和 MAE 改善，建议继续变大'
                                    else:
                                        param_suggestions[param] = '变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，但 RMSE 和 MAE 恶化，建议变小'
                                elif current_val < previous_val:
                                    if current_rmse[j] < previous_rmse[j] and current_mae[j] < previous_mae[j]:
                                        param_suggestions[param] = '继续变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，且 RMSE 和 MAE 改善，建议继续变小'
                                    else:
                                        param_suggestions[param] = '变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，但 RMSE 和 MAE 恶化，建议变大'
                            elif param in ['filters', 'temporal_kernel_size']:
                                if current_val > previous_val:
                                    if current_rmse[j] < previous_rmse[j] and current_mae[j] < previous_mae[j]:
                                        param_suggestions[param] = '继续变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，且 RMSE 和 MAE 改善，建议继续变大'
                                    else:
                                        param_suggestions[param] = '变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，但 RMSE 和 MAE 恶化，建议变小'
                                elif current_val < previous_val:
                                    if current_rmse[j] < previous_rmse[j] and current_mae[j] < previous_mae[j]:
                                        param_suggestions[param] = '继续变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，且 RMSE 和 MAE 改善，建议继续变小'
                                    else:
                                        param_suggestions[param] = '变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，但 RMSE 和 MAE 恶化，建议变大'
                            elif param in ['spatial_kernel_sizes', 'dynamic_attention_units']:
                                if current_val > previous_val:
                                    if current_nse[j] > previous_nse[j]:
                                        param_suggestions[param] = '继续变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，且 NSE 改善，建议继续变大'
                                    else:
                                        param_suggestions[param] = '变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变大，但 NSE 恶化，建议变小'
                                elif current_val < previous_val:
                                    if current_nse[j] > previous_nse[j]:
                                        param_suggestions[param] = '继续变小'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，且 NSE 改善，建议继续变小'
                                    else:
                                        param_suggestions[param] = '变大'
                                        param_suggestions[f'{param}_reason'] = f'当前 {param} ({current_val}) 较前一次 ({previous_val}) 变小，但 NSE 恶化，建议变大'

                # 初始过拟合或欠拟合判断
                if val_loss[-1] > train_loss[-1] + loss_tolerance:
                    param_suggestions['lstm_units'] = '稍稍变小'
                    param_suggestions['lstm_units_reason'] = f'验证损失({val_loss[-1]:.4f})高于训练损失({train_loss[-1]:.4f})且差异超过容忍度({loss_tolerance:.4f})，考虑减少 LSTM 单元数'
                    param_suggestions['dense_units'] = '稍稍变小'
                    param_suggestions['dense_units_reason'] = f'验证损失({val_loss[-1]:.4f})高于训练损失({train_loss[-1]:.4f})且差异超过容忍度({loss_tolerance:.4f})，考虑减少 Dense 单元数'
                elif val_loss[-1] < train_loss[-1] - loss_tolerance:
                    param_suggestions['lstm_units'] = '稍稍变大'
                    param_suggestions['lstm_units_reason'] = f'验证损失({val_loss[-1]:.4f})低于训练损失({train_loss[-1]:.4f})且差异超过容忍度({loss_tolerance:.4f})，考虑增加 LSTM 单元数'
                    param_suggestions['dense_units'] = '稍稍变大'
                    param_suggestions['dense_units_reason'] = f'验证损失({val_loss[-1]:.4f})低于训练损失({train_loss[-1]:.4f})且差异超过容忍度({loss_tolerance:.4f})，考虑增加 Dense 单元数'



                suggestions.append({
                    'target': target,
                    'suggestions': param_suggestions
                })

            

        print(suggestions)

        # Save the suggestions to a JSON file
        suggestions_path = os.path.join('media', str(project_id), f'{model_filename}_suggestions.json')
        with open(suggestions_path, 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, ensure_ascii=False, indent=4)

        return {
            'status': 'success',
            'message': '参数建议生成成功！',
            'suggestions': suggestions
        }

    except Exception as e:
        print(f"Error in analyze_results_and_suggest_params: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    
    






from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args

def train_with_bayesian_optimization(data, target_columns, standard_date_features, time_features, geographical_features, meteorological_features, project_id, model_filename, test_size_ratio=0.2, stop_event=None,n_calls=10,weights={'w1': 0.25, 'w2': 0.25, 'w3': 0.5}):
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
            time_features.extend(['year', 'month', 'day'])

        X_time = data[time_features].values.astype(float)
        X_geographical = data[geographical_features].values.astype(float)
        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)  

        scalers = {}
        for feature_type, X in zip(['time', 'geographical', 'meteorological'], [X_time, X_geographical, X_meteorological]):
            if X.size > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scalers[f'X_{feature_type}'] = (scaler, X_scaled)
            else:
                scalers[f'X_{feature_type}'] = (None, X)

        X_time_scaler, X_time_scaled = scalers['X_time']
        X_geographical_scaler, X_geographical_scaled = scalers['X_geographical']
        X_meteorological_scaler, X_meteorological_scaled = scalers['X_meteorological']

        # 数据准备
        X_time_train, X_time_test, X_geographical_train, X_geographical_test, X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
            X_time_scaled, X_geographical_scaled, X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)

        # 准备数据
        X_time_train = X_time_train.reshape(X_time_train.shape[0], X_time_train.shape[1], 1)  # LSTM 需要 3D 输入
        X_time_test = X_time_test.reshape(X_time_test.shape[0], X_time_test.shape[1], 1)
        X_geographical_train = X_geographical_train.reshape(X_geographical_train.shape[0], X_geographical_train.shape[1], 1)
        X_geographical_test = X_geographical_test.reshape(X_geographical_test.shape[0], X_geographical_test.shape[1], 1)
        X_meteorological_train = X_meteorological_train.reshape(X_meteorological_train.shape[0], X_meteorological_train.shape[1], 1)
        X_meteorological_test = X_meteorological_test.reshape(X_meteorological_test.shape[0], X_meteorological_test.shape[1], 1)

        # 合并输入数据
        X_train = np.concatenate([X_time_train, X_geographical_train, X_meteorological_train], axis=1)
        X_test = np.concatenate([X_time_test, X_geographical_test, X_meteorological_test], axis=1)

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
            Integer(16, 256, name='lstm_units'),
            Integer(16, 256, name='dense_units'),
            Integer(8, 128, name='filters'),
            Integer(2, 128, name='temporal_kernel_size'),
            Integer(2, 128, name='spatial_kernel_size'),  # 修改为单个整数
            Integer(16, 256, name='dynamic_attention_units')
        ]

        @use_named_args(param_space)
        def objective(**params):
            epochs = params['epochs']
            batch_size = params['batch_size']
            lstm_units = params['lstm_units']
            dense_units = params['dense_units']
            filters = params['filters']
            temporal_kernel_size = (params['temporal_kernel_size'],)  # 将整数转换为元组
            spatial_kernel_size = params['spatial_kernel_size']  # 单个整数
            spatial_kernel_sizes = [int(spatial_kernel_size)]  # 将单个整数转换为列表，并确保是普通整数
            dynamic_attention_units = params['dynamic_attention_units']

            # 调试信息
            print(f"epochs: {epochs}")
            print(f"batch_size: {batch_size}")
            print(f"lstm_units: {lstm_units}")
            print(f"dense_units: {dense_units}")
            print(f"filters: {filters}")
            print(f"temporal_kernel_size: {temporal_kernel_size}")
            print(f"spatial_kernel_sizes: {spatial_kernel_sizes}")
            print(f"type(spatial_kernel_sizes): {type(spatial_kernel_sizes)}")
            print(f"spatial_kernel_sizes[0]: {spatial_kernel_sizes[0]}")
            print(f"type(spatial_kernel_sizes[0]): {type(spatial_kernel_sizes[0])}")

            input_shape = X_train.shape[1:]
            time_steps = len(time_features)
            geographical_steps = len(geographical_features)

            model, model_loss = build_temporal_aware_lstm(
                input_shape=input_shape,
                time_steps=time_steps,
                geographical_steps=geographical_steps,
                target_columns=target_columns,
                lstm_units=lstm_units,
                dense_units=dense_units,
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_sizes=spatial_kernel_sizes,  # 使用列表
                dynamic_attention_units=dynamic_attention_units
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            if model_loss is not None:
                history1 = model_loss.fit(
                    X_train,
                    [y_train[:, i] for i in range(len(target_columns))],
                    validation_data=(X_test, [y_test[:, i] for i in range(len(target_columns))]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
            
                y_pred_list = model_loss.predict(X_test)
                y_pred = np.column_stack(y_pred_list)
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )

                y_pred = model.predict(X_test)

                

            # 检查是否触发了 EarlyStopping
            if early_stopping.stopped_epoch == 0:
                # 计算 RMSE, MAE, NSE
                rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0)).tolist()
                mae = np.mean(np.abs(y_pred - y_test), axis=0).tolist()
                nse = 1 - np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)

                # 将 rmse, mae, nse 转换为 numpy 数组
                rmse = np.array(rmse)
                mae = np.array(mae)
                nse = np.array(nse)

                # 计算综合评分
                scores = []
                for i in range(len(target_columns)):
                    if 0 <= nse[i] <= 1:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i] + weights['w3'] * (1 - nse[i]) * 50
                    else:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i]
                    scores.append(score)

                # 取平均值作为最终评分
                score = np.mean(scores)

                return score
            else:
                return 1e6

        class ProgressCallback_b(Callback):
            def __init__(self, total_iterations, project_id, channel_layer):
                super().__init__()
                self.total_iterations = total_iterations
                self.project_id = project_id
                self.channel_layer = channel_layer  # 引入 channel_layer

            def update_progress(self):
                nonlocal current_iteration
                current_iteration += 1
                progress = int((current_iteration / self.total_iterations) * 100)
                print(f"Iteration {current_iteration}/{total_iterations}, Progress: {progress}%")
                try:
                    async_to_sync(self.channel_layer.group_send)(  # 使用 self.channel_layer
                        f'training_progress_{self.project_id}',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'status': '正在缩小参数范围（贝叶斯优化）'
                        }
                    )
                except Exception as e:
                    print(f"Error sending progress: {str(e)}")

        # 使用闭包来处理传递的参数
        def create_callback(progress_callback):
            def callback(result):
                progress_callback.update_progress()
            return callback

        # 贝叶斯优化

        total_iterations = n_calls
        current_iteration = 0

        channel_layer = get_channel_layer()
        progress_callback = ProgressCallback_b(total_iterations, project_id, channel_layer)  # 传递 channel_layer

        res_gp = gp_minimize(objective, param_space, n_calls=n_calls, random_state=0, callback=create_callback(progress_callback))        



        # 重置进度条
        current_iteration = -1
        progress_callback.update_progress()

        # 使用贝叶斯优化找到的最优参数组合构建小范围的参数组合
        best_params = res_gp.x
        print("Best Parameters:", best_params)
        # small_param_grid = {
        #     'epochs': [best_params[0], best_params[0] + 1,best_params[0] + 2],
        #     'batch_size': [best_params[1] - 1, best_params[1], best_params[1] + 1],
        #     'lstm_units': [best_params[2] - 1, best_params[2], best_params[2] + 1],
        #     'dense_units': [best_params[3] - 1, best_params[3], best_params[3] + 1],
        #     'filters': [best_params[4] - 1, best_params[4], best_params[4] + 1],
        #     'temporal_kernel_size': [(best_params[5],)],  # 只包含最优参数本身
        #     'spatial_kernel_sizes': [[int(best_params[6])]],  # 只包含最优参数本身
        #     'dynamic_attention_units': [best_params[7] - 1, best_params[7], best_params[7] + 1]
        # }
        small_param_grid = {
            'epochs': [ best_params[0]],
            'batch_size': [best_params[1]],
            'lstm_units': [ best_params[2]],
            'dense_units': [ best_params[3], best_params[3] + 1],
            'filters': [ best_params[4], best_params[4] + 1],
            'temporal_kernel_size': [(best_params[5],)],  # 只包含最优参数本身
            'spatial_kernel_sizes': [[int(best_params[6])]],  # 只包含最优参数本身
            'dynamic_attention_units': [ best_params[7]]
        }


        results = []
        param_names = list(small_param_grid.keys())
        param_values = [small_param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        total_iterations = len(param_combinations)
        current_iteration=0

        channel_layer = get_channel_layer()
        
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

         # 小范围参数组合训练进度条
        input_shape = X_train.shape[1:]
        time_steps = len(time_features)
        geographical_steps = len(geographical_features)

        # 在训练循环中，确保在训练结束后调用 update_progress 方法
        best_result = None
        best_score = float('inf')

        for params in param_combinations:
            if stop_event and stop_event.is_set():
                print("Stop event detected. Stopping training.")
                break

            
            param_dict = dict(zip(param_names, params))
            print('param_dict', param_dict)
            epochs = param_dict.pop('epochs')
            batch_size = param_dict.pop('batch_size')

            
            start_time = datetime.now()

            progress_callback = ProgressCallback(total_iterations, epochs, project_id)  

            model, model_loss = build_temporal_aware_lstm(
                input_shape=input_shape,
                time_steps=time_steps,
                geographical_steps=geographical_steps,
                target_columns=target_columns,
                **param_dict
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            if model_loss is not None:
                history1 = model_loss.fit(
                    X_train,
                    [y_train[:, i] for i in range(len(target_columns))],
                    validation_data=(X_test, [y_test[:, i] for i in range(len(target_columns))]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, progress_callback],
                    verbose=0
                )
                
                y_pred_list = model_loss.predict(X_test)
                y_pred = np.column_stack(y_pred_list)
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, progress_callback],
                    verbose=0
                )

                y_pred = model.predict(X_test)

            # 在训练结束后调用 update_progress 方法
            progress_callback.update_progress()

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            # 检查是否触发了 EarlyStopping
            if early_stopping.stopped_epoch == 0:
                rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0)).tolist()
                mae = np.mean(np.abs(y_pred - y_test), axis=0).tolist()
                nse = (1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))).tolist()
                r2 = (1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))).tolist()
                adj_r2 = (1 - (1 - np.array(r2)) * (len(y_test) - 1) / (len(y_test) - len(meteorological_features) - 1 - 1)).tolist()
                
                # 将 rmse, mae, nse 转换为 numpy 数组
                rmse = np.array(rmse)
                mae = np.array(mae)
                nse = np.array(nse)

                # 计算综合评分
                scores = []
                for i in range(len(target_columns)):
                    if 0 <= nse[i] <= 1:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i] + weights['w3'] * (1 - nse[i]) * 50
                    else:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i]
                    scores.append(score)

                # 取平均值作为最终评分
                score = np.mean(scores)

                if model_loss is not None:
                    results.append({
                        'params': param_dict,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'adj_r2': adj_r2,
                        'nse': nse,
                        'training_time': training_time,
                        'history': history1.history,
                    })
                else:
                    results.append({
                        'params': param_dict,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'adj_r2': adj_r2,
                        'nse': nse,
                        'training_time': training_time,
                        'history': history.history,
                    })

                # 更新最优结果
                if score < best_score:
                    best_score = score
                    best_result = {
                        'params': param_dict,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'adj_r2': adj_r2,
                        'nse': nse,
                        'training_time': training_time,
                        'history': history.history if model_loss is None else history1.history,
                        'model': model if model_loss is None else model_loss,
                        'score': score
                    }

            else:
                print("EarlyStopping triggered. Skipping this combination.")

        

        if not results:
            print("No parameter combinations completed all epochs without early stopping.")
            return {
                'status': 'error',
                'message': '所有参数组合均触发了 EarlyStopping，未找到合适的参数组合。'
            }

        
        # 保存最优模型
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
            'best_params': best_result_data['best_params'],
            'best_model_filename': model_filename
        }

    except Exception as e:
        print(f"Error in train_with_bayesian_optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }



def train_with_bayesian(data, target_columns, standard_date_features, time_features, geographical_features, meteorological_features, project_id, test_size_ratio=0.2, n_calls=10,weights={'w1': 0.25, 'w2': 0.25, 'w3': 0.5}):
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
            time_features.extend(['year', 'month', 'day'])

        X_time = data[time_features].values.astype(float)
        X_geographical = data[geographical_features].values.astype(float)
        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)  

        scalers = {}
        for feature_type, X in zip(['time', 'geographical', 'meteorological'], [X_time, X_geographical, X_meteorological]):
            if X.size > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scalers[f'X_{feature_type}'] = (scaler, X_scaled)
            else:
                scalers[f'X_{feature_type}'] = (None, X)

        X_time_scaler, X_time_scaled = scalers['X_time']
        X_geographical_scaler, X_geographical_scaled = scalers['X_geographical']
        X_meteorological_scaler, X_meteorological_scaled = scalers['X_meteorological']

        # 数据准备
        X_time_train, X_time_test, X_geographical_train, X_geographical_test, X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
            X_time_scaled, X_geographical_scaled, X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)

        # 准备数据
        X_time_train = X_time_train.reshape(X_time_train.shape[0], X_time_train.shape[1], 1)  # LSTM 需要 3D 输入
        X_time_test = X_time_test.reshape(X_time_test.shape[0], X_time_test.shape[1], 1)
        X_geographical_train = X_geographical_train.reshape(X_geographical_train.shape[0], X_geographical_train.shape[1], 1)
        X_geographical_test = X_geographical_test.reshape(X_geographical_test.shape[0], X_geographical_test.shape[1], 1)
        X_meteorological_train = X_meteorological_train.reshape(X_meteorological_train.shape[0], X_meteorological_train.shape[1], 1)
        X_meteorological_test = X_meteorological_test.reshape(X_meteorological_test.shape[0], X_meteorological_test.shape[1], 1)

        # 合并输入数据
        X_train = np.concatenate([X_time_train, X_geographical_train, X_meteorological_train], axis=1)
        X_test = np.concatenate([X_time_test, X_geographical_test, X_meteorological_test], axis=1)

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
            epochs_range = (10, 70)
            batch_size_range = (16, 128)
        else:
            epochs_range = (5, 50)
            batch_size_range = (32, 256)

        param_space = [
            Integer(epochs_range[0], epochs_range[1], name='epochs'),
            Integer(batch_size_range[0], batch_size_range[1], name='batch_size'),
            Integer(16, 256, name='lstm_units'),
            Integer(16, 256, name='dense_units'),
            Integer(8, 128, name='filters'),
            Integer(2, 128, name='temporal_kernel_size'),
            Integer(2, 128, name='spatial_kernel_size'),  # 修改为单个整数
            Integer(16, 256, name='dynamic_attention_units')
        ]

        best_score = 1e6

        @use_named_args(param_space)
        def objective(**params):
            nonlocal best_score
            epochs = params['epochs']
            batch_size = params['batch_size']
            lstm_units = params['lstm_units']
            dense_units = params['dense_units']
            filters = params['filters']
            temporal_kernel_size = (params['temporal_kernel_size'],)  # 将整数转换为元组
            spatial_kernel_size = params['spatial_kernel_size']  # 单个整数
            spatial_kernel_sizes = [int(spatial_kernel_size)]  # 将单个整数转换为列表，并确保是普通整数
            dynamic_attention_units = params['dynamic_attention_units']

            # 调试信息
            print(f"epochs: {epochs}")
            print(f"batch_size: {batch_size}")
            print(f"lstm_units: {lstm_units}")
            print(f"dense_units: {dense_units}")
            print(f"filters: {filters}")
            print(f"temporal_kernel_size: {temporal_kernel_size}")
            print(f"spatial_kernel_sizes: {spatial_kernel_sizes}")
            print(f"type(spatial_kernel_sizes): {type(spatial_kernel_sizes)}")
            print(f"spatial_kernel_sizes[0]: {spatial_kernel_sizes[0]}")
            print(f"type(spatial_kernel_sizes[0]): {type(spatial_kernel_sizes[0])}")

            input_shape = X_train.shape[1:]
            time_steps = len(time_features)
            geographical_steps = len(geographical_features)

            model, model_loss = build_temporal_aware_lstm(
                input_shape=input_shape,
                time_steps=time_steps,
                geographical_steps=geographical_steps,
                target_columns=target_columns,
                lstm_units=lstm_units,
                dense_units=dense_units,
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_sizes=spatial_kernel_sizes,  # 使用列表
                dynamic_attention_units=dynamic_attention_units
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            if model_loss is not None:
                history1 = model_loss.fit(
                    X_train,
                    [y_train[:, i] for i in range(len(target_columns))],
                    validation_data=(X_test, [y_test[:, i] for i in range(len(target_columns))]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
            
                y_pred_list = model_loss.predict(X_test)
                y_pred = np.column_stack(y_pred_list)
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )

                y_pred = model.predict(X_test)

            # 检查是否触发了 EarlyStopping
            if early_stopping.stopped_epoch == 0:
                # 计算 RMSE, MAE, NSE
                rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0)).tolist()
                mae = np.mean(np.abs(y_pred - y_test), axis=0).tolist()
                nse = 1 - np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)

                # 将 rmse, mae, nse 转换为 numpy 数组
                rmse = np.array(rmse)
                mae = np.array(mae)
                nse = np.array(nse)

                # 计算综合评分
                scores = []
                for i in range(len(target_columns)):
                    if 0 <= nse[i] <= 1:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i] + weights['w3'] * (1 - nse[i]) * 50
                    else:
                        score = weights['w1'] * rmse[i] + weights['w2'] * mae[i]
                    scores.append(score)

                # 取平均值作为最终评分
                score = np.mean(scores)
                if best_score>=score:
                    best_score = score

                return score
            else:
                return 1e6

        class ProgressCallback_b(Callback):
            def __init__(self, total_iterations, project_id, channel_layer):
                super().__init__()
                self.total_iterations = total_iterations
                self.project_id = project_id
                self.channel_layer = channel_layer  # 引入 channel_layer

            def update_progress(self):
                nonlocal current_iteration
                current_iteration += 1
                progress = int((current_iteration / self.total_iterations) * 100)
                print(f"Iteration {current_iteration}/{total_iterations}, Progress: {progress}%")
                try:
                    async_to_sync(self.channel_layer.group_send)(  # 使用 self.channel_layer
                        f'training_progress_{self.project_id}',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'status': '正在缩小参数范围（贝叶斯优化）'
                        }
                    )
                except Exception as e:
                    print(f"Error sending progress: {str(e)}")

        # 使用闭包来处理传递的参数
        def create_callback(progress_callback):
            def callback(result):
                progress_callback.update_progress()
            return callback

        # 贝叶斯优化
        
        total_iterations = n_calls
        current_iteration = 0
        

        channel_layer = get_channel_layer()
        progress_callback = ProgressCallback_b(total_iterations, project_id, channel_layer)  # 传递 channel_layer

        res_gp = gp_minimize(objective, param_space, n_calls=n_calls, random_state=0, callback=create_callback(progress_callback))        



        # 重置进度条
        current_iteration = -1
        progress_callback.update_progress()

        # 使用贝叶斯优化找到的最优参数组合构建小范围的参数组合
        best_params = res_gp.x
        best_params = [int(item) for item in best_params]
        print("Best Parameters:", best_params)

        return {
            'status': 'success',
            'message': '模型训练并保存成功！',
            'best_params': best_params,
            'score': best_score
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
    best_result_data['rmse'] = [float(v) if isinstance(v, np.float32) else v for v in best_result_data['rmse']]
    best_result_data['mae'] = [float(v) if isinstance(v, np.float32) else v for v in best_result_data['mae']]
    best_result_data['r2'] = [float(v) if isinstance(v, np.float32) else v for v in best_result_data['r2']]
    best_result_data['adj_r2'] = [float(v) if isinstance(v, np.float32) else v for v in best_result_data['adj_r2']]
    best_result_data['nse'] = [float(v) if isinstance(v, np.float32) else v for v in best_result_data['nse']]

    # 处理 training_time
    if isinstance(best_result_data['training_time'], np.float64):
        best_result_data['training_time'] = float(best_result_data['training_time'])

    return best_result_data







def train_multiple_param_combinations(data, target_columns, standard_date_features, time_features, geographical_features, meteorological_features, param_grid, project_id, test_size_ratio=0.2, weights=None):
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
            time_features.extend(['year', 'month', 'day'])

        X_time = data[time_features].values.astype(float)
        X_geographical = data[geographical_features].values.astype(float)
        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
        y = data[target_columns].values.astype(float)

        scalers = {}
        for feature_type, X in zip(['time', 'geographical', 'meteorological'], [X_time, X_geographical, X_meteorological]):
            if X.size > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scalers[f'X_{feature_type}'] = (scaler, X_scaled)
            else:
                scalers[f'X_{feature_type}'] = (None, X)

        X_time_scaler, X_time_scaled = scalers['X_time']
        X_geographical_scaler, X_geographical_scaled = scalers['X_geographical']
        X_meteorological_scaler, X_meteorological_scaled = scalers['X_meteorological']

        # 数据准备
        X_time_train, X_time_test, X_geographical_train, X_geographical_test, X_meteorological_train, X_meteorological_test, y_train, y_test = train_test_split(
            X_time_scaled, X_geographical_scaled, X_meteorological_scaled, y, test_size=test_size_ratio, shuffle=True, random_state=42)

        # 准备数据
        X_time_train = X_time_train.reshape(X_time_train.shape[0], X_time_train.shape[1], 1)  # LSTM 需要 3D 输入
        X_time_test = X_time_test.reshape(X_time_test.shape[0], X_time_test.shape[1], 1)
        X_geographical_train = X_geographical_train.reshape(X_geographical_train.shape[0], X_geographical_train.shape[1], 1)
        X_geographical_test = X_geographical_test.reshape(X_geographical_test.shape[0], X_geographical_test.shape[1], 1)
        X_meteorological_train = X_meteorological_train.reshape(X_meteorological_train.shape[0], X_meteorological_train.shape[1], 1)
        X_meteorological_test = X_meteorological_test.reshape(X_meteorological_test.shape[0], X_meteorological_test.shape[1], 1)

        # 合并输入数据
        X_train = np.concatenate([X_time_train, X_geographical_train, X_meteorological_train], axis=1)
        X_test = np.concatenate([X_time_test, X_geographical_test, X_meteorological_test], axis=1)


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

        

        input_shape = X_train.shape[1:]
        time_steps = len(time_features)
        geographical_steps = len(geographical_features)

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            epochs = param_dict.pop('epochs')
            batch_size = param_dict.pop('batch_size')

            print('param_dict', param_dict)
            start_time = datetime.now()

            progress_callback = ProgressCallback(total_iterations, epochs, project_id)

            model, model_loss = build_temporal_aware_lstm(
                input_shape=input_shape,
                time_steps=time_steps,
                geographical_steps=geographical_steps,
                target_columns=target_columns,
                **param_dict
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            if model_loss is not None:
                history1 = model_loss.fit(
                    X_train,
                    [y_train[:, i] for i in range(len(target_columns))],
                    validation_data=(X_test, [y_test[:, i] for i in range(len(target_columns))]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, progress_callback],
                    verbose=0
                )
            
                y_pred_list = model_loss.predict(X_test)
                y_pred = np.column_stack(y_pred_list)
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, progress_callback],
                    verbose=0
                )

                y_pred = model.predict(X_test)

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0)).tolist()
            mae = np.mean(np.abs(y_pred - y_test), axis=0).tolist()
            nse = (1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))).tolist()
            r2 = (1 - (np.sum((y_pred - y_test) ** 2, axis=0) / np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0))).tolist()
            adj_r2 = (1 - (1 - np.array(r2)) * (len(y_test) - 1) / (len(y_test) - len(meteorological_features) - 1 - 1)).tolist()
            
            # 计算 AIC 和 BIC
            num_samples = y_test.shape[0]
            num_params = model.count_params()
            mse = np.mean((y_pred - y_test) ** 2)
            aic = num_samples * np.log(mse) + 2 * num_params
            bic = num_samples * np.log(mse) + num_params * np.log(num_samples)
            
            if model_loss is not None:
                results.append({
                    'params': param_dict,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'adj_r2': adj_r2,
                    'nse': nse,
                    'training_time': training_time,
                    'aic': aic,
                    'bic': bic
                })
            else:
                results.append({
                    'params': param_dict,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'adj_r2': adj_r2,
                    'nse': nse,
                    'training_time': training_time,
                    'aic': aic,
                    'bic': bic
                })

        print(results)
        

        if weights:
            evaluate_models(results, weights, project_id, target_columns)

        return {
            'status': 'success',
            'message': '模型训练并保存成功！',
            'results': results
        }

    except Exception as e:
        print(f"Error in train_multiple_param_combinations: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


def evaluate_models(results, weights, project_id, target_columns):
    try:
        calculated_results = []
        
        for result in results:
            single_result = {}
            
            # 拆解 params 字典
            for key, value in result['params'].items():
                if isinstance(value, list):
                    value = ','.join(map(str, value))
                single_result[key] = value
            
            # 添加其他字段
            single_result['epochs'] = result['epochs']
            single_result['batch_size'] = result['batch_size']
            
            # 添加 AIC 和 BIC
            single_result['aic'] = result['aic']
            single_result['bic'] = result['bic']
            
            num_targets = len(result['rmse'])

            
            for i, target in enumerate(target_columns):
                avg_rmse = result['rmse'][i]
                avg_mae = result['mae'][i]
                avg_nse = result['nse'][i]
                
                single_result[f'rmse_{target}'] = avg_rmse
                single_result[f'mae_{target}'] = avg_mae
                single_result[f'r2_{target}'] = result['r2'][i]
                single_result[f'adj_r2_{target}'] = result['adj_r2'][i]
                single_result[f'nse_{target}'] = avg_nse
                
                single_result[f'score_{target}'] = (
                    weights['w1'] * avg_rmse +
                    weights['w2'] * avg_mae +
                    weights['w3'] * (1 - avg_nse)*50
                )
            
             


            single_result['training_time'] = result['training_time']
            
            calculated_results.append(single_result)

        
        results_df = pd.DataFrame(calculated_results)
        csv_file_path = os.path.join('media', str(project_id), f'calculate_result.csv')
        results_df.to_csv(csv_file_path, index=False)

        print("Evaluation results saved to calculate_result.csv")

        return {
            'status': 'success',
            'message': '模型评估成功！',
            'results': calculated_results
        }

    except Exception as e:
        print(f"Error in evaluate_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


    

def predict_model(data, target_columns, standard_date_features, time_features, geographical_features, meteorological_features, project_id, train_df):
    try:
        
        # 定义自定义层
        custom_objects = {
            'TemporalAwareConv1D': TemporalAwareConv1D,
            'SpatialAwareConv1D': SpatialAwareConv1D,
            'TimeAwareAttention': TimeAwareAttention,
            'PositionalEncoding': PositionalEncoding,
            'DynamicAttention': DynamicAttention,
            'WeightedFusion': WeightedFusion,
            'r_square': r_square
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
            time_features.extend(['year', 'month', 'day'])

            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            data['year'] = data[date_column].dt.year
            data['month'] = data[date_column].dt.month
            data['day'] = data[date_column].dt.day

        X_train_time = train_df[time_features].values.astype(float)
        X_train_geographical = train_df[geographical_features].values.astype(float)
        X_train_meteorological = train_df[meteorological_features].values.astype(float) if meteorological_features else np.empty((train_df.shape[0], 0))

        scalers = {}
        for feature_type, X_train in zip(['time', 'geographical', 'meteorological'], [X_train_time, X_train_geographical, X_train_meteorological]):
            if X_train.size > 0:
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_scaled = scaler.transform(X_train)
                scalers[f'X_{feature_type}'] = (scaler, X_scaled)
            else:
                scalers[f'X_{feature_type}'] = (None, X_train)

        X_time_scaler, _ = scalers['X_time']
        X_geographical_scaler, _ = scalers['X_geographical']
        X_meteorological_scaler, _ = scalers['X_meteorological']

        X_time = data[time_features].values.astype(float)
        X_geographical = data[geographical_features].values.astype(float)
        X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))

        X_time_scaled = X_time_scaler.transform(X_time) if X_time_scaler else X_time
        X_geographical_scaled = X_geographical_scaler.transform(X_geographical) if X_geographical_scaler else X_geographical
        X_meteorological_scaled = X_meteorological_scaler.transform(X_meteorological) if X_meteorological_scaler else X_meteorological

        # 准备数据
        X_time_scaled = X_time_scaled.reshape(X_time_scaled.shape[0], X_time_scaled.shape[1], 1)  # LSTM 需要 3D 输入
        X_geographical_scaled = X_geographical_scaled.reshape(X_geographical_scaled.shape[0], X_geographical_scaled.shape[1], 1)
        X_meteorological_scaled = X_meteorological_scaled.reshape(X_meteorological_scaled.shape[0], X_meteorological_scaled.shape[1], 1)

        # 合并输入数据
        X_scaled = np.concatenate([X_time_scaled, X_geographical_scaled, X_meteorological_scaled], axis=1)

        print(f"X_scaled shape: {X_scaled.shape}")

        y_pred = model.predict(X_scaled)

        if len(target_columns) == 1:
            data[f'predict_{target_columns[0]}'] = y_pred.flatten()
        else:
            y_pred_list = model.predict(X_scaled)
            y_pred = np.column_stack(y_pred_list)
            for i, target in enumerate(target_columns):
                data[f'predict_{target}'] = y_pred[:, i]

        # Save the predicted data
        predicted_csv_file_path = os.path.join('media', str(project_id), f'predicted_result.csv')
        data.to_csv(predicted_csv_file_path, index=False)

        return {
            'status': 'success',
            'message': '预测完成并保存成功！',
            'predicted_file': f'predicted_result.csv'
        }

    except Exception as e:
        print(f"Error in predict_model: {str(e)}")
        import traceback
        traceback.print_exc()  
        return {
            'status': 'error',
            'message': str(e)
        }
    






def plot_loss_web(results, project_id, model_filename, num_outputs, target_columns):
    # Use D3.js to render the loss chart
    # Example:
    loss_data = []
    for result in results:
        history = result['history']
        for j in range(num_outputs):
            train_key = f'output_{j}_loss' if num_outputs > 1 else 'loss'
            val_key = f'val_output_{j}_loss' if num_outputs > 1 else 'val_loss'
            # Convert numpy.float32 to float
            train_loss = [float(loss) for loss in history[train_key]]
            val_loss = [float(loss) for loss in history[val_key]]
            loss_data.append({
                'target': target_columns[j],
                'train_loss': train_loss,
                'val_loss': val_loss
            })

    with open(os.path.join('media', str(project_id), f'{model_filename}_loss.json'), 'w') as f:
        json.dump(loss_data, f)

def plot_metrics_web(results, project_id, model_filename, num_outputs, target_columns):
    # Use D3.js to render the metrics chart
    # Example:
    metrics_data = []
    for result in results:
        metrics_data.append({
            'target': target_columns,
            'rmse': result['rmse'],
            'mae': result['mae'],
            'r2': result['r2'],
            'adj_r2': result['adj_r2'],
            'nse': result['nse'],
            
            'time': [result['training_time']] * num_outputs
        })

    with open(os.path.join('media', str(project_id), f'{model_filename}_metrics.json'), 'w') as f:
        json.dump(metrics_data, f)

def plot_predictions_web(y_true, y_pred, project_id, model_filename, num_outputs, target_columns):
    # Convert y_true and y_pred to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    predictions_data = []
    for j in range(num_outputs):
        predictions_data.append({
            'target': target_columns[j],
            'y_true': y_true[:, j].tolist(),
            'y_pred': y_pred[:, j].tolist()
        })

    with open(os.path.join('media', str(project_id), f'{model_filename}_predictions.json'), 'w') as f:
        json.dump(predictions_data, f)

def plot_feature_importance_web(scaler_time, scaler_geographical, scaler_meteorological,
                             X_time, X_geographical, X_meteorological,
                             X_time_scaled, X_geographical_scaled, X_meteorological_scaled,
                             project_id, model_filename, time_features, geographical_features, meteorological_features):
    # Ensure X_time_scaled, X_geographical_scaled, and X_meteorological_scaled are NumPy arrays
    if not isinstance(X_time_scaled, np.ndarray):
        X_time_scaled = np.array(X_time_scaled)
    if not isinstance(X_geographical_scaled, np.ndarray):
        X_geographical_scaled = np.array(X_geographical_scaled)
    if not isinstance(X_meteorological_scaled, np.ndarray):
        X_meteorological_scaled = np.array(X_meteorological_scaled)

    # Combine all features and their scaled means
    all_features = np.concatenate([X_time, X_geographical, X_meteorological], axis=1)
    all_features_scaled = np.concatenate([X_time_scaled, X_geographical_scaled, X_meteorological_scaled], axis=1)
    
    feature_names = time_features + geographical_features + meteorological_features
    
    original_feature_names = feature_names  # Use provided feature names

    scaled_feature_means = all_features_scaled.mean(axis=0)
    feature_importances = np.argsort(scaled_feature_means)[::-1]  # Ensure feature_importances is an array

    print(f"Original Feature Names: {original_feature_names}")
    print(f"Scaled Feature Means: {scaled_feature_means}")
    print(f"Feature Importances: {feature_importances}")

    # Prepare feature importance data for JSON
    feature_importance_data = []
    for i, feature in enumerate(feature_names):
        feature_importance_data.append({
            'feature': feature,
            'importance': scaled_feature_means[i]
        })

    # Save feature importance data to JSON file
    plot_path = os.path.join('media', str(project_id), f'{model_filename}_feature_importance.json')
    with open(plot_path, 'w') as f:
        json.dump(feature_importance_data, f)



def preprocess_data(data, time_features, geographical_features, meteorological_features, standard_date_features):
    # 处理标准日期特征
    if standard_date_features:
        date_column = standard_date_features[0]
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(by=date_column)
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        data['day'] = data[date_column].dt.day
        time_features.extend(['year', 'month', 'day'])

    # 提取特征和目标变量
    X_time = data[time_features].values.astype(float)
    X_geographical = data[geographical_features].values.astype(float)
    X_meteorological = data[meteorological_features].values.astype(float) if meteorological_features else np.empty((data.shape[0], 0))
    
    # 标准化特征
    scalers = {}
    for feature_type, X in zip(['time', 'geographical', 'meteorological'], [X_time, X_geographical, X_meteorological]):
        if X.size > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scalers[f'X_{feature_type}'] = (scaler, X_scaled)
        else:
            scalers[f'X_{feature_type}'] = (None, X)

    X_time_scaler, X_time_scaled = scalers['X_time']
    X_geographical_scaler, X_geographical_scaled = scalers['X_geographical']
    X_meteorological_scaler, X_meteorological_scaled = scalers['X_meteorological']

    return X_time_scaled, X_geographical_scaled, X_meteorological_scaled, scalers




def plot_histogram_web(data, features, project_id, model_filename):
    histograms = {}
    histograms_2d = {}
    histograms_3d = {}

    # Process time features
    if 'standard_date_feature' in features:
        date_column = features['standard_date_feature'][0]
        data[date_column] = pd.to_datetime(data[date_column])
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        data['day'] = data[date_column].dt.day
        time_features = list(set(features['time_feature'] + ['year', 'month', 'day']))
    else:
        time_features = list(set(features['time_feature'])) 

    for feature in time_features:
        feature_data = data[feature].value_counts().to_dict()
        histograms['time'] = histograms.get('time', {})
        histograms['time'][feature] = feature_data

    # Process geographical features
    geographical_features = list(set(features['geographical_feature']))  # 去重处理

    for feature in geographical_features:
        feature_data = data[feature].value_counts().to_dict()
        histograms['geographical'] = histograms.get('geographical', {})
        histograms['geographical'][feature] = feature_data

    # Ensure all values are numbers
    for key, feature_dict in histograms.items():
        for feature, counts in feature_dict.items():
            histograms[key][feature] = {k: float(v) for k, v in counts.items()}

    # Generate 2D histograms
    for i in range(len(time_features)):
        for j in range(len(time_features)):
            feature1 = time_features[i]
            feature2 = time_features[j]
            key = f"{feature1}_vs_{feature2}"
            histograms_2d['time'] = histograms_2d.get('time', {})
            histograms_2d['time'][key] = data.groupby([feature1, feature2]).size().unstack(fill_value=0).to_dict()

    for i in range(len(geographical_features)):
        for j in range(len(geographical_features)):
            feature1 = geographical_features[i]
            feature2 = geographical_features[j]
            key = f"{feature1}_vs_{feature2}"
            histograms_2d['geographical'] = histograms_2d.get('geographical', {})
            histograms_2d['geographical'][key] = data.groupby([feature1, feature2]).size().unstack(fill_value=0).to_dict()

    # Generate 3D histograms
    for i in range(len(time_features)):
        for j in range(i + 1, len(time_features)):
            for k in range(j + 1, len(time_features)):
                feature1 = time_features[i]
                feature2 = time_features[j]
                feature3 = time_features[k]
                key = f"{feature1}_vs_{feature2}_vs_{feature3}"
                histograms_3d['time'] = histograms_3d.get('time', {})
                histograms_3d['time'][key] = data.groupby([feature1, feature2, feature3]).size().unstack(fill_value=0).unstack(fill_value=0).to_dict()

    for i in range(len(geographical_features)):
        for j in range(i + 1, len(geographical_features)):
            for k in range(j + 1, len(geographical_features)):
                feature1 = geographical_features[i]
                feature2 = geographical_features[j]
                feature3 = geographical_features[k]
                key = f"{feature1}_vs_{feature2}_vs_{feature3}"
                histograms_3d['geographical'] = histograms_3d.get('geographical', {})
                histograms_3d['geographical'][key] = data.groupby([feature1, feature2, feature3]).size().unstack(fill_value=0).unstack(fill_value=0).to_dict()

    # Convert all keys in histograms_3d to strings
    histograms_3d = {str(k): v for k, v in histograms_3d.items()}
    for key in histograms_3d:
        histograms_3d[key] = {str(k2): v2 for k2, v2 in histograms_3d[key].items()}
        for key2 in histograms_3d[key]:
            histograms_3d[key][key2] = {str(k3): v3 for k3, v3 in histograms_3d[key][key2].items()}

    plot_path = os.path.join('media', str(project_id), f'{model_filename}_histograms.json')
    with open(plot_path, 'w') as f:
        json.dump(histograms, f)

    plot_path_2d = os.path.join('media', str(project_id), f'{model_filename}_2d_histograms.json')
    with open(plot_path_2d, 'w') as f:
        json.dump(histograms_2d, f)

    plot_path_3d = os.path.join('media', str(project_id), f'{model_filename}_3d_histograms.json')
    with open(plot_path_3d, 'w') as f:
        json.dump(histograms_3d, f)


def plot_parallel_coordinates_web(data, features, project_id, model_filename):
    selected_features = features['other_feature'] + features['target_variable']
    
    tg_features = features['time_feature'] + features['geographical_feature']
    
    plot_data = []
    for _, row in data.iterrows():
        entry = {
            'target_variable': {col: row[col] for col in features['target_variable']},
            'other_feature': {col: row[col] for col in selected_features if col not in features['target_variable']},
            'tg_feature': {col: row[col] for col in tg_features}
        }
        plot_data.append(entry)

    plot_path = os.path.join('media', str(project_id), f'{model_filename}_parallel_coordinates.json')
    with open(plot_path, 'w') as f:
        json.dump(plot_data, f)

