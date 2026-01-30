function renderRadarChart(data, previousData) {
    console.log("Rendering radar chart with data:", data);

    // 获取 radarChartContainer 元素
    const chartDom = document.getElementById('radarChartContainer');
    if (!chartDom) {
        console.error("radarChartContainer element not found");
        return;
    }

    // 初始化 ECharts 实例
    const myChart = echarts.init(chartDom);

    // 处理数据
    if (!data || data.length === 0) {
        console.error("No data provided for radar chart");
        return;
    }

    // 假设 data 是一个数组，包含一个对象，该对象包含多个目标变量的指标
    const firstEntry = data[0];
    const indicators = ['rmse', 'mae', 'r2', 'adj_r2', 'nse','time'];
    const seriesData = [];

    // 构建雷达图的指标
    const radarIndicator = indicators.map(indicator => {
        const currentValues = firstEntry[indicator];
        let previousValues = [];
        if (previousData && previousData.length > 0) {
            const previousFirstEntry = previousData[0];
            previousValues = previousFirstEntry[indicator] || [];
        }

        const combinedValues = [...currentValues, ...previousValues];

        if (!combinedValues || combinedValues.length === 0) {
            console.error(`No values for indicator ${indicator}`);
            return { name: indicator.toUpperCase(), max: 1, min: 0 };
        }

        const max = Math.max(...combinedValues) > 1 ? Math.max(...combinedValues) * 1.1 : Math.max(...combinedValues) < 0 ? Math.max(...combinedValues) * 0.9 : 1;
        const min = Math.min(...combinedValues) < 0 ? Math.min(...combinedValues) * 1.1 : Math.min(...combinedValues) > 1 ? Math.min(...combinedValues) * 0.9 : 0;

        return {
            name: indicator.toUpperCase(),
            max: max,
            min: min
        };
    });

    // 构建雷达图的数据
    firstEntry.target.forEach((target, index) => {
        const seriesItem = {
            value: indicators.map(indicator => {
                const value = firstEntry[indicator][index];
                return value !== undefined ? value : 0; // 确保值存在
            }),
            name: target,
            itemStyle: {
                color: defaultColors[index % defaultColors.length] 
            }
        };
        seriesData.push(seriesItem);
    });

    // 处理前一次训练的结果数据
    if (previousData && previousData.length > 0) {
        const previousFirstEntry = previousData[0];
        previousFirstEntry.target.forEach((target, index) => {
            const seriesItem = {
                value: indicators.map(indicator => {
                    const value = previousFirstEntry[indicator][index];
                    return value !== undefined ? value : 0; // 确保值存在
                }),
                name: `Previous ${target}`,
                itemStyle: {
                    color: previousColors[index % previousColors.length] 
                }
            };
            seriesData.push(seriesItem);
        });
    }

    // 配置项
    const option = {
        radar: {
            indicator: radarIndicator,
            shape: 'circle',
            splitNumber: 4,
            axisName: {
                formatter: '{value}',
                color: '#fff',
                backgroundColor: '#999',
                borderRadius: 3,
                padding: [3, 5]
            },
            axisLabel: { // 添加 axisLabel 配置
                show: true, // 确保轴标签显示
                formatter: function (value, index) {
                    return value.toFixed(1); // 保留一位小数
                },
                color: '#666', // 设置字体颜色更淡
                fontSize: 10 // 设置字体更小
            }
        },
        tooltip: { // 添加 tooltip 配置
            trigger: 'item',
            formatter: function (params) {
                let tooltip = `<strong>${params.seriesName}</strong><br>`;
                params.value.forEach((value, index) => {
                    tooltip += `${indicators[index].toUpperCase()}: ${value.toFixed(2)}<br>`;
                });
                return tooltip;
            }
        },
        legend: { // 添加 legend 配置
            data: seriesData.map(item => item.name),
            bottom: 0,
            left: 'center',
            selectedMode: 'multiple'
        },
        series: [{
            name: 'Metrics',
            type: 'radar',
            data: seriesData,
            areaStyle: {
                color: 'rgba(150, 150, 150, 0.1)'
            }
        }]
        
    };

    myChart.setOption(option);
}





function renderLossChart(data, previousData) {
    console.log("Rendering loss chart with data:", data);

    // 获取 lossChartSVG 元素
    const chartDom = document.getElementById('lossChartSVG');
    if (!chartDom) {
        console.error("lossChartSVG element not found");
        return;
    }

    // 初始化 ECharts 实例
    const myChart = echarts.init(chartDom);

    // 处理数据
    if (!data || data.length === 0) {
        console.error("No data provided for loss chart");
        return;
    }

    // 构建图例和系列数据
    const legendData = [];
    const seriesData = [];

    // 处理当前训练的数据
    data.forEach((entry, index) => {
        const target = entry.target;

        // train_loss 系列
        legendData.push({
            name: `Train Loss ${target}`,
            icon: 'line', // 使用线条作为图例图标
            textStyle: { color: defaultColors[index % defaultColors.length] }
        });

        seriesData.push({
            name: `Train Loss ${target}`,
            type: 'line',
            data: entry.train_loss.map(value => parseFloat(value.toFixed(1))),
            smooth: true,
            symbol: 'none', // 不显示符号点
            lineStyle: {
                width: 2,
                color: defaultColors[index % defaultColors.length]
            },
        });

        // val_loss 系列
        if (entry.val_loss) {
            legendData.push({
                name: `Val Loss ${target}`,
                icon: 'line', // 使用线条作为图例图标
                textStyle: { color: lightColors[index % lightColors.length] }
            });

            seriesData.push({
                name: `Val Loss ${target}`,
                type: 'line',
                data: entry.val_loss.map(value => parseFloat(value.toFixed(1))),
                smooth: true,
                symbol: 'none', // 不显示符号点
                lineStyle: {
                    width: 2,
                    color: lightColors[index % lightColors.length]
                }
            });
        }
    });

    // 处理前一次训练的数据
    if (previousData && previousData.length > 0) {
        previousData.forEach((entry, index) => {
            const target = entry.target;

            // Previous train_loss 系列
            legendData.push({
                name: `Previous Train Loss ${target}`,
                icon: 'line', // 使用线条作为图例图标
                textStyle: { color: previousColors[index % previousColors.length] }
            });

            seriesData.push({
                name: `Previous Train Loss ${target}`,
                type: 'line',
                data: entry.train_loss.map(value => parseFloat(value.toFixed(1))),
                smooth: true,
                symbol: 'none', // 不显示符号点
                lineStyle: {
                    width: 2,
                    color: previousColors[index % previousColors.length]
                }
            });

            // Previous val_loss 系列
            if (entry.val_loss) {
                legendData.push({
                    name: `Previous Val Loss ${target}`,
                    icon: 'line', // 使用线条作为图例图标
                    textStyle: { color: previousLightColors[index % previousLightColors.length] }
                });

                seriesData.push({
                    name: `Previous Val Loss ${target}`,
                    type: 'line',
                    data: entry.val_loss.map(value => parseFloat(value.toFixed(1))),
                    smooth: true,
                    symbol: 'none', // 不显示符号点
                    lineStyle: {
                        width: 2,
                        color: previousLightColors[index % previousLightColors.length]
                    }
                });
            }
        });
    }

    // 配置项
    const option = {
        legend: {
            data: legendData,
            bottom: 10,
            left: 'center',
            selectedMode: 'multiple',
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'line'
            },
            formatter: function (params) {
                let tooltip = `<strong>Epoch ${params[0].axisValue}</strong><br>`;
                params.forEach(param => {
                    // 找到对应的 legendData 条目
                    const legendItem = legendData.find(item => item.name === param.seriesName);
                    const color = legendItem ? legendItem.textStyle.color : '#000'; // 默认颜色为黑色
                    tooltip += `<span style="color:${color}; font-size:20px; line-height:20px;">●</span> ${param.seriesName}: ${param.value.toFixed(2)}<br>`;
                });
                return tooltip;
            }
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: data[0].train_loss.map((_, index) => `Epoch ${index + 1}`)
        },
        yAxis: {
            type: 'value',
            name: 'Loss',
            axisLabel: {
                formatter: function (value) {
                    return parseFloat(value.toFixed(1));
                }
            }
        },
        series: seriesData
    };

    myChart.setOption(option);
}





function renderImportanceChart(data) {
    console.log("Rendering importance chart with data:", data);

    // 获取 importanceChartSVG 元素
    const chartDom = document.getElementById('importanceChartSVG');
    if (!chartDom) {
        console.error("importanceChartSVG element not found");
        return;
    }

    // 销毁之前的 ECharts 实例（如果存在）
    const existingChart = echarts.getInstanceByDom(chartDom);
    if (existingChart) {
        existingChart.dispose();
    }

    // 初始化 ECharts 实例
    const myChart = echarts.init(chartDom);

    // 处理数据
    if (!data || data.length === 0) {
        console.error("No data provided for importance chart");
        return;
    }

    // 按照 importance 值降序排序
    data.sort((a, b) => b.importance - a.importance);

    const labels = data.map(d => d.feature);
    const values = data.map(d => parseFloat((d.importance * Math.pow(10, 16)).toFixed(1))); // 乘以10的16次方并保留一位小数

    // 使用 colorPalette 设置颜色
    const colors = data.map((_, index) => colorPalette[index % colorPalette.length]);

    // 配置项
    const option = {
        yAxis: {
            type: 'category',
            data: labels,
            inverse: true, // 设置为 true 以使数据从上到下排列
            axisLabel: {
                rotate: 0,
                interval: 0,
                margin: 10 // 增加标签与轴的距离
            },
            axisTick: {
                show: true
            },
            axisLine: {
                show: true
            },
            position: 'left' // 确保Y轴在左侧
        },
        xAxis: {
            type: 'value',
            show: true, // 显示X轴
            name: 'Importance',
            axisLabel: {
                formatter: function (value) {
                    return parseFloat(value.toFixed(1));
                }
            }
        },
        series: [{
            name: 'Importance',
            type: 'bar',
            data: values,
            itemStyle: {
                color: function(params) {
                    return colors[params.dataIndex];
                }
            }
        }],
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            width: '80%', // 调整宽度以适应现有宽度
            left: '20%', // 增加左侧间距以适应Y轴标签
            top: '3%', // 减少顶部空白
            bottom: '3%' // 减少底部空白
        }
    };

    myChart.setOption(option);
}


function renderPredictionChart(data, previousData) {
    console.log("Rendering prediction chart with data:", data);
    const chartDom = document.getElementById('predictionChartSVG');
    // 清除之前的图表内容
    chartDom.innerHTML = '';
    if (!chartDom) {
        console.error("predictionChartSVG element not found");
        return;
    }

    // 销毁之前的 ECharts 实例（如果存在）
    const existingChart = echarts.getInstanceByDom(chartDom);
    if (existingChart) {
        existingChart.dispose();
    }

    

    // 初始化 ECharts 实例
    const myChart = echarts.init(chartDom);

    

    if (!data || data.length === 0) {
        console.error("No data provided for prediction chart");
        return;
    }

    const seriesData = [];
    const legendData = [];
    let labels = []; // 定义 labels 变量

    data.forEach((targetData, index) => {
        labels = targetData.y_true.map((_, index) => index + 1); // 更新 labels 变量
        const actualColor = lightColors[index];
        const predictedColor = defaultColors[index];

        seriesData.push({
            name: `Actual ${targetData.target}`,
            type: 'line',
            data: targetData.y_true,
            smooth: true,
            symbol: 'circle',
            symbolSize: 8,
            showSymbol: false,
            lineStyle: {
                width: 2,
                color: actualColor
            },
            itemStyle: {
                color: actualColor
            },
            // Only show the first target variable initially
            show: index === 0
        });

        seriesData.push({
            name: `Predicted ${targetData.target}`,
            type: 'line',
            data: targetData.y_pred,
            smooth: true,
            symbol: 'circle',
            symbolSize: 8,
            showSymbol: false,
            lineStyle: {
                width: 2,
                color: predictedColor
            },
            itemStyle: {
                color: predictedColor
            },
            // Only show the first target variable initially
            show: index === 0
        });

        legendData.push({
            name: `Actual ${targetData.target}`,
            icon: 'circle',
            textStyle: { color: actualColor }
        });

        legendData.push({
            name: `Predicted ${targetData.target}`,
            icon: 'circle',
            textStyle: { color: predictedColor }
        });
    });

    // 处理前一次训练的结果数据
    if (previousData && previousData.length > 0) {
        previousData.forEach((targetData, index) => {
            const actualColor = previousLightColors[index];
            const predictedColor = previousColors[index];

            seriesData.push({
                name: `Previous Actual ${targetData.target}`,
                type: 'line',
                data: targetData.y_true,
                smooth: true,
                symbol: 'circle',
                symbolSize: 8,
                showSymbol: false,
                lineStyle: {
                    width: 2,
                    color: actualColor
                },
                itemStyle: {
                    color: actualColor
                },
                // Only show the first target variable initially
                show: index === 0
            });

            seriesData.push({
                name: `Previous Predicted ${targetData.target}`,
                type: 'line',
                data: targetData.y_pred,
                smooth: true,
                symbol: 'circle',
                symbolSize: 8,
                showSymbol: false,
                lineStyle: {
                    width: 2,
                    color: predictedColor
                },
                itemStyle: {
                    color: predictedColor
                },
                // Only show the first target variable initially
                show: index === 0
            });

            legendData.push({
                name: `Previous Actual ${targetData.target}`,
                icon: 'circle',
                textStyle: { color: actualColor }
            });

            legendData.push({
                name: `Previous Predicted ${targetData.target}`,
                icon: 'circle',
                textStyle: { color: predictedColor }
            });
        });
    }

    const chartData = {
        xAxis: {
            type: 'category',
            data: labels, // 使用 labels 变量
            name: 'Sample',
            axisLabel: {
                formatter: function (value) {
                    return value;
                }
            }
        },
        yAxis: {
            type: 'value',
            name: 'Value',
            axisLabel: {
                formatter: function (value) {
                    return parseFloat(value.toFixed(1));
                }
            }
        },
        series: seriesData,
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'line'
            }
        },
        legend: {
            data: legendData,
            bottom: 10,
            left: 'center',
            itemWidth: 10, // 设置图例项的宽度
            itemHeight: 10, // 设置图例项的高度
            itemStyle: {
                color: function(params) {
                    const index = legendData.findIndex(legendItem => legendItem.name === params.name);
                    return index % 2 === 0 ? (index < data.length * 2 ? lightColors[Math.floor(index / 2)] : previousLightColors[Math.floor((index - data.length * 2) / 2)]) : (index < data.length * 2 ? defaultColors[Math.floor(index / 2)] : previousColors[Math.floor((index - data.length * 2) / 2)]);
                }
            },
            selected: legendData.reduce((acc, legendItem, index) => {
                // Only select the first target variable initially
                if (index < 2) {
                    acc[legendItem.name] = true;
                } else {
                    acc[legendItem.name] = false;
                }
                return acc;
            }, {})
        }
    };

    myChart.setOption(chartData);
}

function toggleHistogram(type) {
    const canvasId = 'timeHistogramCanvas'; // 使用同一个 canvas 元素
    const legendId = 'timeLegend'; // 使用同一个 legend 元素

    if (type === 'time') {
        document.getElementById(canvasId).style.display = 'block';
        renderHistogram(canvasId, timeData, timeData2d, timeData3d);
    } else if (type === 'geo') {
        document.getElementById(canvasId).style.display = 'block';
        renderHistogram(canvasId, geoData, geoData2d, geoData3d);
    }
}


function handleTargetClick(target) {
    toggleRadar(target);
    toggleLoss(target);
    togglePrediction(target);
}

function toggleRadar(target) {
    console.log("Toggle RadarChart for target:", target);

    // 获取雷达图的 ECharts 实例
    const radarChartContainer = document.getElementById('radarChartContainer');
    const myChart = echarts.getInstanceByDom(radarChartContainer);

    if (myChart) {
        // 获取当前的 legend 数据
        const legendData = myChart.getOption().legend[0].data;

        // 找到与目标变量对应的 legend 索引
        const targetIndex = legendData.indexOf(target);

        if (targetIndex !== -1) {
            // 获取当前 legend 项的状态
            const legendSelected = myChart.getOption().legend[0].selected;
            const isSelected = legendSelected[target];

            // 根据当前状态选择或取消选择 legend 项
            try {
                if (isSelected) {
                    myChart.dispatchAction({
                        type: 'legendUnSelect',
                        name: target
                    });
                    updateSpanColor(target, false); // 更新 span 颜色为灰色
                } else {
                    myChart.dispatchAction({
                        type: 'legendSelect',
                        name: target
                    });
                    updateSpanColor(target, true); // 更新 span 颜色为正常颜色
                }
            } catch (error) {
                console.error("Error dispatching legend action:", error);
            }
        }
    }
}

function toggleLoss(target) {
    console.log("Toggle Loss for target:", target);

    const lossChartSVG = document.getElementById('lossChartSVG');
    const myChart = echarts.getInstanceByDom(lossChartSVG);

    if (myChart) {
        const legendData = myChart.getOption().legend[0].data;
        console.log("legendData:", legendData);
        // 确保 legendData 是一个数组
        if (Array.isArray(legendData)) {
            // 构建 train_loss 和 val_loss 的目标名称
            let trainLossTarget = `Train Loss ${target}`;
            let valLossTarget = `Val Loss ${target}`;

            // 如果目标名称包含 "Previous"，则相应地构建 train_loss 和 val_loss 的目标名称
            if (target.startsWith('Previous ')) {
                trainLossTarget = `Previous Train Loss ${target.split(' ')[1]}`;
                valLossTarget = `Previous Val Loss ${target.split(' ')[1]}`;
            }

            // 找到与目标变量对应的 "Train Loss" 和 "Val Loss" 的 legend 索引
            const trainLossTargetIndex = legendData.findIndex(item => item.name.toLowerCase() === trainLossTarget.toLowerCase());
            const valLossTargetIndex = legendData.findIndex(item => item.name.toLowerCase() === valLossTarget.toLowerCase());

            console.log("Train Loss Target Index:", trainLossTargetIndex); // 调试：输出 trainLossTargetIndex
            console.log("Val Loss Target Index:", valLossTargetIndex); // 调试：输出 valLossTargetIndex

            if (trainLossTargetIndex !== -1 || valLossTargetIndex !== -1) {
                // 获取当前 legend 项的状态
                const legendSelected = myChart.getOption().legend[0].selected;
                console.log("Legend Selected:", legendSelected); // 调试：输出 legendSelected

                const trainLossIsSelected = legendSelected[trainLossTarget];
                const valLossIsSelected = legendSelected[valLossTarget];
                console.log("Train Loss Is Selected:", trainLossIsSelected); // 调试：输出 trainLossIsSelected
                console.log("Val Loss Is Selected:", valLossIsSelected); // 调试：输出 valLossIsSelected

                // 根据当前状态选择或取消选择 legend 项
                try {
                    if (trainLossIsSelected && valLossIsSelected) {
                        myChart.dispatchAction({
                            type: 'legendUnSelect',
                            name: trainLossTarget
                        });
                        if (valLossTargetIndex !== -1) {
                            myChart.dispatchAction({
                                type: 'legendUnSelect',
                                name: valLossTarget
                            });
                        }
                        console.log("Unselected Train and Val Loss for target:", target); // 调试：输出取消选择的信息
                    } else {
                        myChart.dispatchAction({
                            type: 'legendSelect',
                            name: trainLossTarget
                        });
                        if (valLossTargetIndex !== -1) {
                            myChart.dispatchAction({
                                type: 'legendSelect',
                                name: valLossTarget
                            });
                        }
                        console.log("Selected Train and Val Loss for target:", target); // 调试：输出选择的信息
                    }
                } catch (error) {
                    console.error("Error dispatching legend action:", error);
                }
            } else {
                console.error("Legend entries for Train Loss or Val Loss not found for target:", target); // 调试：输出未找到 legend 条目的信息
            }
        } else {
            console.error("Legend data is not an array:", legendData);
        }
    } else {
        console.error("ECharts instance not found for loss chart");
    }
}

function updateSpanColor(target, isSelected) {
    const spans = document.querySelectorAll('.parallel-coordinates h4 span');
    spans.forEach(span => {
        if (span.getAttribute('data-target') === target) {
            if (isSelected) {
                // 检查目标变量是否包含 "Previous" 前缀
                if (target.startsWith('Previous ')) {
                    const index = Array.from(spans).indexOf(span);
                    span.style.color = previousColors[(index-2) % previousColors.length]; // 设置 previousColors 颜色
                } else {
                    const index = Array.from(spans).indexOf(span);
                    span.style.color = defaultColors[index % defaultColors.length]; // 设置 defaultColors 颜色
                }
            } else {
                span.style.color = 'gray'; // 设置灰色
            }
        }
    });
}

function togglePrediction(target) {
    console.log("Toggle Prediction for target:", target);

    // 获取预测图的 ECharts 实例
    const predictionChartSVG = document.getElementById('predictionChartSVG');
    const myChart = echarts.getInstanceByDom(predictionChartSVG);

    if (myChart) {
        // 获取当前的 legend 数据
        const legendData = myChart.getOption().legend[0].data;
        console.log("Legend Data:", legendData); // 调试：输出 legend 数据

        // 构建实际和预测的目标名称
        let actualTarget = `Actual ${target}`;
        let predictedTarget = `Predicted ${target}`;

        // 如果目标名称包含 "Previous"，则相应地构建实际和预测的目标名称
        if (target.startsWith('Previous ')) {
            actualTarget = `Previous Actual ${target.split(' ')[1]}`;
            predictedTarget = `Previous Predicted ${target.split(' ')[1]}`;
        }

        // 找到与目标变量对应的 "Actual" 和 "Predicted" 的 legend 索引
        const actualTargetIndex = legendData.findIndex(item => item.name.toLowerCase() === actualTarget.toLowerCase());
        const predictedTargetIndex = legendData.findIndex(item => item.name.toLowerCase() === predictedTarget.toLowerCase());

        console.log("Actual Target Index:", actualTargetIndex); // 调试：输出 actualTargetIndex
        console.log("Predicted Target Index:", predictedTargetIndex); // 调试：输出 predictedTargetIndex

        if (actualTargetIndex !== -1 && predictedTargetIndex !== -1) {
            // 获取当前 legend 项的状态
            const legendSelected = myChart.getOption().legend[0].selected;
            console.log("Legend Selected:", legendSelected); // 调试：输出 legendSelected

            const actualIsSelected = legendSelected[actualTarget];
            const predictedIsSelected = legendSelected[predictedTarget];
            console.log("Actual Is Selected:", actualIsSelected); // 调试：输出 actualIsSelected
            console.log("Predicted Is Selected:", predictedIsSelected); // 调试：输出 predictedIsSelected

            // 根据当前状态选择或取消选择 legend 项
            try {
                if (actualIsSelected && predictedIsSelected) {
                    myChart.dispatchAction({
                        type: 'legendUnSelect',
                        name: actualTarget
                    });
                    myChart.dispatchAction({
                        type: 'legendUnSelect',
                        name: predictedTarget
                    });
                    console.log("Unselected Actual and Predicted for target:", target); // 调试：输出取消选择的信息
                } else {
                    myChart.dispatchAction({
                        type: 'legendSelect',
                        name: actualTarget
                    });
                    myChart.dispatchAction({
                        type: 'legendSelect',
                        name: predictedTarget
                    });
                    console.log("Selected Actual and Predicted for target:", target); // 调试：输出选择的信息
                }
            } catch (error) {
                console.error("Error dispatching legend action:", error);
            }
        } else {
            console.error("Legend entries for Actual or Predicted not found for target:", target); // 调试：输出未找到 legend 条目的信息
        }
    } else {
        console.error("ECharts instance not found for prediction chart"); // 调试：输出未找到 ECharts 实例的信息
    }
}





function showErrorMessage(message) {
    const errorMessageDiv = document.getElementById('errorMessage');
    errorMessageDiv.textContent = message;
    errorMessageDiv.style.display = 'block';
}

function clearErrorMessage() {
    const errorMessageDiv = document.getElementById('errorMessage');
    errorMessageDiv.textContent = '';
    errorMessageDiv.style.display = 'none';
}