def analyze_results_and_suggest_params(results, project_id, model_filename, num_outputs, target_columns,triggered_epoch=None):
    try:
        # Load previous results if they exist
        previous_results_path = os.path.join('media', str(project_id), f'{model_filename}_metrics.json')
        if os.path.exists(previous_results_path):
            with open(previous_results_path, 'r') as f:
                previous_results = json.load(f)
        else:
            previous_results = []

        suggestions = []

        for i, result in enumerate(results):
            current_rmse = result['rmse']
            current_mae = result['mae']
            current_r2 = result['r2']
            current_adj_r2 = result['adj_r2']
            current_nse = result['nse']
            current_training_time = result['training_time']
            current_params = result['params']

            # Compare with previous results if available
            if previous_results:
                previous_rmse = previous_results[i]['rmse']
                previous_mae = previous_results[i]['mae']
                previous_r2 = previous_results[i]['r2']
                previous_adj_r2 = previous_results[i]['adj_r2']
                previous_nse = previous_results[i]['nse']
                previous_training_time = previous_results[i]['time']


            param_suggestions = {}

            if previous_results:    
                for j, target in enumerate(target_columns):
                    time_increase = (current_training_time - previous_training_time[j])/previous_training_time[j]

                    # 分析训练和验证损失
                    train_loss = result['train_loss']
                    val_loss = result['val_loss']

                    # 定义容忍度阈值
                    loss_tolerance = 1.5


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
                    else:
                        # 如果差异在容忍度范围内，使用 R² 和 Adj R² 的判断
                        if current_r2[j] < 0.7 or current_adj_r2[j] < 0.7:
                            # 如果当前 R2 或 Adj R2 小于 0.7，建议增加 LSTM 和 Dense 层数
                            param_suggestions['lstm_units'] = '变大'
                            param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 小于 0.7，考虑增加 LSTM 单元数'
                            param_suggestions['dense_units'] = '变大'
                            param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 小于 0.7，考虑增加 Dense 单元数'
                        elif current_r2[j] < 0.9 or current_adj_r2[j] < 0.9:
                            # 如果当前 R2 或 Adj R2 在 0.7 到 0.9 之间，建议稍微增加 LSTM 和 Dense 层数
                            param_suggestions['lstm_units'] = '稍稍变大'
                            param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加 LSTM 单元数'
                            param_suggestions['dense_units'] = '稍稍变大'
                            param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加 Dense 单元数'
                        else:
                            if (current_r2[j] - previous_r2[j]) < 0:
                                # 如果 R2 变差，建议稍微减少 LSTM 和 Dense 层数
                                param_suggestions['lstm_units'] = '稍稍变小'
                                param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f})相较于前一次当前 R² ({previous_r2[j]:.4f}) 变差，考虑稍微减少 LSTM 单元数'
                                param_suggestions['dense_units'] = '稍稍变小'
                                param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f})相较于前一次当前 R² ({previous_r2[j]:.4f}) 变差，考虑稍微减少 Dense 单元数'
                            else:
                                # 如果当前 R2 在 0.9-1.0 之间，建议稍微增加 LSTM 和 Dense 层数
                                param_suggestions['lstm_units'] = '稍稍变大或不变'
                                param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变 LSTM 单元数'
                                param_suggestions['dense_units'] = '稍稍变大或不变'
                                param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变 Dense 单元数'

                    if triggered_epoch is not None:
                            # 优先判断早停轮次，并建议 epochs 取值为比早停伦次小的值
                            param_suggestions['epochs'] = '稍稍变小'
                            param_suggestions['epochs_reason'] = f'早停触发于第 {triggered_epoch} 轮，建议将 epochs 减少到小于{triggered_epoch}的数值'    



                    # NSE
                    if current_nse[j] < 0.7:
                        # 如果当前 NSE 小于 0.7，建议增加卷积核大小、动态注意力单位数和训练轮数
                        param_suggestions['spatial_kernel_sizes'] = '变大'
                        param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加卷积核大小'
                        param_suggestions['dynamic_attention_units'] = '变大'
                        param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加动态注意力单位数'
                        if triggered_epoch is None:
                            param_suggestions['epochs'] = '变大'
                            param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加训练轮数'
                    elif current_nse[j] < 0.9:
                        # 如果当前 NSE 在 0.7 到 0.9 之间，建议稍微增加卷积核大小、动态注意力单位数和训练轮数
                        param_suggestions['spatial_kernel_sizes'] = '稍稍变大'
                        param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加卷积核大小'
                        param_suggestions['dynamic_attention_units'] = '稍稍变大'
                        param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加动态注意力单位数'
                        if triggered_epoch is None:    
                            param_suggestions['epochs'] = '稍稍变大'
                            param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加训练轮数'
                    else:
                        # 如果 NSE 改善不大且训练时间增长过大，建议稍微减少卷积核大小、动态注意力单位数和训练轮数
                        if (current_nse[j] - previous_nse[j]) < 0:
                            param_suggestions['spatial_kernel_sizes'] = '稍稍变小'
                            param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f})相较于前一次NSE ({previous_nse[j]:.4f}) 变差，考虑稍微减少卷积核大小'
                            param_suggestions['dynamic_attention_units'] = '稍稍变小'
                            param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f})相较于前一次NSE ({previous_nse[j]:.4f}) 变差，考虑稍微减少动态注意力单位数'
                            if triggered_epoch is None:    
                                param_suggestions['epochs'] = '稍稍变大'
                                param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f})相较于前一次NSE ({previous_nse[j]:.4f}) 变差，考虑稍微增大训练轮数'
                        else:
                            param_suggestions['spatial_kernel_sizes'] = '稍稍变大或不变'
                            param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变卷积核大小'
                            param_suggestions['dynamic_attention_units'] = '稍稍变大或不变'
                            param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变动态注意力单位数'
                            if triggered_epoch is None:
                                param_suggestions['epochs'] = '稍稍变大或不变'
                                param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.9-1.0 之间，考虑稍微增加或不变训练轮数'

                    # Filters and Temporal Kernel Size
                    if current_rmse[j] > 3.0 or current_mae[j] > 2.5:
                        # 如果当前 RMSE 或 MAE 大于，建议增加滤波器数量和时间卷积核大小
                        param_suggestions['filters'] = '变大'
                        param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑增加滤波器数量'
                        param_suggestions['temporal_kernel_size'] = '变大'
                        param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑增加时间卷积核大小'
                    elif current_rmse[j] > 1.5 or current_mae[j] > 1.0:
                        # 如果当前 RMSE 或 MAE 在 到 之间，建议稍微增加滤波器数量和时间卷积核大小
                        param_suggestions['filters'] = '稍稍变大'
                        param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 在 1.5 到 3 或 1.0 到 2.5 之间，考虑稍微增加滤波器数量'
                        param_suggestions['temporal_kernel_size'] = '稍稍变大'
                        param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 在 1.5 到 3 或 1.0 到 2.5 之间，考虑稍微增加时间卷积核大小'
                    elif (previous_rmse[j] - current_rmse[j]) < 0 or (previous_mae[j] - current_mae[j]) < 0 :
                        # 如果 RMSE 改善不大且训练时间增长过大，建议稍微减少滤波器数量和时间卷积核大小
                        param_suggestions['filters'] = '稍稍变小'
                        param_suggestions['temporal_kernel_size'] = '稍稍变小'
                        if (previous_rmse[j] - current_rmse[j]) < 0:
                            param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f})相较于前一次RMSE ({current_rmse[j]:.4f}) 变差，考虑稍微减少滤波器数量'                           
                            param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f})相较于前一次RMSE ({current_rmse[j]:.4f}) 变差，考虑稍微减少时间卷积核大小'
                        elif (previous_mae[j] - current_mae[j]) < 0:
                            param_suggestions['filters_reason'] = f'当前 MAE ({current_mae[j]:.4f})相较于前一次MAE ({current_mae[j]:.4f}) 变差，考虑稍微减少滤波器数量'
                            param_suggestions['temporal_kernel_size_reason'] = f'当前 MAE ({current_mae[j]:.4f})相较于前一次MAE ({current_mae[j]:.4f}) 变差，考虑稍微减少时间卷积核大小'
                        else:
                            param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f}) 前一次RMSE ({current_rmse[j]:.4f})和前一次MAE ({current_mae[j]:.4f})变差，考虑稍微减少滤波器数量'
                            param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f})相较于前一次RMSE ({current_rmse[j]:.4f})和前一次MAE ({current_mae[j]:.4f})变差，考虑稍微减少时间卷积核大小'

                    else:
                        # 否则稍稍变大或不变
                        param_suggestions['filters'] = '稍稍变大或不变'
                        param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f}) 改善良好，考虑稍微增加或不变滤波器数量'
                        param_suggestions['temporal_kernel_size'] = '稍稍变大或不变'
                        param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 和 MAE ({current_mae[j]:.4f}) 改善良好，考虑稍微增加或不变时间卷积核大小'

                    # Batch Size
                    if train_loss[-1] > val_loss[-1] and val_loss[-1] < 3:
                        # 如果训练过程稳定且验证损失较低，考虑增大 batch_size
                        param_suggestions['batch_size'] = '稍稍变大'
                        param_suggestions['batch_size_reason'] = f'训练过程稳定且验证损失较低 ({val_loss[-1]:.4f})，考虑增大 batch_size'
                    elif train_loss[-1] < 3 and val_loss[-1] < 3:
                        # 如果训练损失和验证损失均较低且稳定，考虑增大 batch_size
                        param_suggestions['batch_size'] = '稍稍变大'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 均较低且稳定，考虑增大 batch_size'
                    elif train_loss[-1] > val_loss[-1] and val_loss[-1] > 3:
                        # 如果训练损失高于验证损失且验证损失较高，考虑减小 batch_size
                        param_suggestions['batch_size'] = '稍稍变小'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 高于验证损失 ({val_loss[-1]:.4f}) 且验证损失较高，考虑减小 batch_size'
                    else:
                        # 否则保持不变
                        param_suggestions['batch_size'] = '保持不变'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 差异不大，考虑保持 batch_size 不变'

                    suggestions.append({
                        'target': target,
                        'suggestions': param_suggestions
                    })

            else:
                for j, target in enumerate(target_columns):
                    # 分析训练和验证损失
                    train_loss = result['train_loss']
                    val_loss = result['val_loss']

                    # 定义容忍度阈值
                    loss_tolerance = 1.5  # 可以根据实际情况调整

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
                    else:
                        # 如果差异在容忍度范围内，使用 R² 和 Adj R² 的判断
                        if current_r2[j] < 0.7 or current_adj_r2[j] < 0.7:
                            # 如果当前 R2 或 Adj R2 小于 0.7，建议增加 LSTM 和 Dense 层数
                            param_suggestions['lstm_units'] = '变大'
                            param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 小于 0.7，考虑增加 LSTM 单元数'
                            param_suggestions['dense_units'] = '变大'
                            param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 小于 0.7，考虑增加 Dense 单元数'
                        elif current_r2[j] < 0.9 or current_adj_r2[j] < 0.9:
                            # 如果当前 R2 或 Adj R2 在 0.7 到 0.9 之间，建议稍微增加 LSTM 和 Dense 层数
                            param_suggestions['lstm_units'] = '稍稍变大'
                            param_suggestions['lstm_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加 LSTM 单元数'
                            param_suggestions['dense_units'] = '稍稍变大'
                            param_suggestions['dense_units_reason'] = f'当前 R² ({current_r2[j]:.4f}) 或 Adj R² ({current_adj_r2[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加 Dense 单元数'

                    if triggered_epoch is not None:
                            # 优先判断早停轮次，并建议 epochs 取值为比早停伦次小的值
                            param_suggestions['epochs'] = '稍稍变小'
                            param_suggestions['epochs_reason'] = f'早停触发于第 {triggered_epoch} 轮，建议将 epochs 减少到小于{triggered_epoch}的数值'    



                    # NSE
                    if current_nse[j] < 0.7:
                        # 如果当前 NSE 小于 0.7，建议增加卷积核大小、动态注意力单位数和训练轮数
                        param_suggestions['spatial_kernel_sizes'] = '变大'
                        param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加卷积核大小'
                        param_suggestions['dynamic_attention_units'] = '变大'
                        param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加动态注意力单位数'
                        if triggered_epoch is None:
                            param_suggestions['epochs'] = '变大'
                            param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 小于 0.7，考虑增加训练轮数'
                    elif current_nse[j] < 0.9:
                        # 如果当前 NSE 在 0.7 到 0.9 之间，建议稍微增加卷积核大小、动态注意力单位数和训练轮数
                        param_suggestions['spatial_kernel_sizes'] = '稍稍变大'
                        param_suggestions['spatial_kernel_sizes_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加卷积核大小'
                        param_suggestions['dynamic_attention_units'] = '稍稍变大'
                        param_suggestions['dynamic_attention_units_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加动态注意力单位数'
                        if triggered_epoch is None:    
                            param_suggestions['epochs'] = '稍稍变大'
                            param_suggestions['epochs_reason'] = f'当前 NSE ({current_nse[j]:.4f}) 在 0.7 到 0.9 之间，考虑稍微增加训练轮数'

                    # Filters and Temporal Kernel Size
                    if current_rmse[j] > 3.0 or current_mae[j] > 2.5:
                        # 如果当前 RMSE 或 MAE 大于 0.5，建议增加滤波器数量和时间卷积核大小
                        param_suggestions['filters'] = '变大'
                        param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑增加滤波器数量'
                        param_suggestions['temporal_kernel_size'] = '变大'
                        param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 大于 3.0 或 2.5，考虑增加时间卷积核大小'
                    elif current_rmse[j] > 1.5 or current_mae[j] > 1.0:
                        # 如果当前 RMSE 或 MAE 在 0.2 到 0.5 之间，建议稍微增加滤波器数量和时间卷积核大小
                        param_suggestions['filters'] = '稍稍变大'
                        param_suggestions['filters_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 在 1.5 到 3 或 1.0 到 2.5 之间，考虑稍微增加滤波器数量'
                        param_suggestions['temporal_kernel_size'] = '稍稍变大'
                        param_suggestions['temporal_kernel_size_reason'] = f'当前 RMSE ({current_rmse[j]:.4f}) 或 MAE ({current_mae[j]:.4f}) 在 1.5 到 3 或 1.0 到 2.5 之间，考虑稍微增加时间卷积核大小'

                    # Batch Size
                    if train_loss[-1] > val_loss[-1] and val_loss[-1] < 3:
                        # 如果训练过程稳定且验证损失较低，考虑增大 batch_size
                        param_suggestions['batch_size'] = '稍稍变大'
                        param_suggestions['batch_size_reason'] = f'训练过程稳定且验证损失较低 ({val_loss[-1]:.4f})，考虑增大 batch_size'
                    elif train_loss[-1] < 3 and val_loss[-1] < 3:
                        # 如果训练损失和验证损失均较低且稳定，考虑增大 batch_size
                        param_suggestions['batch_size'] = '稍稍变大'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 均较低且稳定，考虑增大 batch_size'
                    elif train_loss[-1] > val_loss[-1] and val_loss[-1] > 3:
                        # 如果训练损失高于验证损失且验证损失较高，考虑减小 batch_size
                        param_suggestions['batch_size'] = '稍稍变小'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 高于验证损失 ({val_loss[-1]:.4f}) 且验证损失较高，考虑减小 batch_size'
                    else:
                        # 否则保持不变
                        param_suggestions['batch_size'] = '保持不变'
                        param_suggestions['batch_size_reason'] = f'训练损失 ({train_loss[-1]:.4f}) 和验证损失 ({val_loss[-1]:.4f}) 差异不大，考虑保持 batch_size 不变'


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
    