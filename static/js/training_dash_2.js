function renderHistogram(canvasId, data, data2d, data3d) {
    console.log("Rendering each histograms:", data);

    // Convert keys to numbers for month and day
    const processedData = {};
    for (let key in data) {
        if (typeof data[key] === 'object' && !Array.isArray(data[key])) {
            processedData[key] = Object.keys(data[key]).map(category => ({
                category: parseFloat(category),
                value: data[key][category]
            }));
        } else {
            console.error(`Invalid data structure for key ${key}:`, data[key]);
        }
    }

    console.log("Processed data:", processedData);

    const keys = Object.keys(processedData);
    if (keys.length === 0) return;

    // Ensure the legend and canvas elements are available
    const legend = document.getElementById(canvasId).parentNode.querySelector(".histogram-legend");
    const canvas = document.getElementById(canvasId);
    if (!legend || !canvas) {
        console.error("Legend or canvas element not found for canvasId:", canvasId);
        return;
    }

    // Initialize legend
    legend.innerHTML = '';

    // Define a color palette with at least 20 colors
    

    // Assign a unique color to each key
    const keyColors = {};
    keys.forEach((key, i) => {
        keyColors[key] = colorPalette[i % colorPalette.length];
    });

    keys.forEach((key, i) => {
        const legendItem = document.createElement("div");
        legendItem.className = "legend-item";
        legendItem.style.display = "flex";
        legendItem.style.alignItems = "center";
        legendItem.style.marginBottom = "5px";

        const legendColor = document.createElement("div");
        legendColor.style.width = "18px";
        legendColor.style.height = "18px";
        legendColor.style.backgroundColor = i === 0 ? keyColors[key] : Chart.helpers.color(keyColors[key]).alpha(0.3).rgbString(); // 修改此处
        legendColor.style.marginRight = "10px";

        const legendLabel = document.createElement("span");
        legendLabel.textContent = key;

        legendItem.appendChild(legendColor);
        legendItem.appendChild(legendLabel);
        legend.appendChild(legendItem);
    });

    // Track selected keys
    const selectedKeys = new Set([keys[0]]);
    console.log("selectedKeys:", selectedKeys);

    // Render initial histogram
    renderSelectedHistogram(canvasId, processedData, selectedKeys, keyColors, data2d, data3d);

    // Add click event to legend items
    const legendItems = legend.querySelectorAll(".legend-item");
    if (legendItems.length === 0) {
        console.error("No legend items found for canvasId:", canvasId);
        return;
    }

    legendItems.forEach((item, index) => {
        item.addEventListener("click", function() {
            const key = keys[index];
            const colorDiv = item.querySelector("div");
            if (!colorDiv) {
                console.error("No div found in legend item for canvasId:", canvasId);
                return;
            }
            console.log("colorDiv found:", colorDiv); // 添加日志信息
            const color = keyColors[key];
            if (selectedKeys.has(key)) {
                selectedKeys.delete(key);
                colorDiv.style.backgroundColor = Chart.helpers.color(color).alpha(0.3).rgbString(); // Set to light version
            } else {
                if (selectedKeys.size < 3) {
                    selectedKeys.add(key);
                    colorDiv.style.backgroundColor = color; // Set to full color
                } else {
                    alert("最多选择三个特征");
                }
            }
            renderSelectedHistogram(canvasId, processedData, selectedKeys, keyColors, data2d, data3d);
        });
    });
}



function renderHistograms(data, data2d, data3d) {
console.log("Rendering histograms:", data);

// Initial render for time histogram
renderHistogram('timeHistogramCanvas', timeData, timeData2d, timeData3d);
}

let currentChartInstances = {}; // 用于存储每个 canvas 的 Chart 实例



let initialCanvasWidth;
let initialCanvasHeight;

let originalCanvasElement = null;
let originalCanvasParentNode = null;
let originalCanvasNextSibling = null;

function renderSelectedHistogram(canvasId, data, selectedKeys, keyColors, data2d, data3d) {
console.log("renderSelectedHistogram called with:", { canvasId, data, selectedKeys, data2d, data3d });

if (selectedKeys.size === 0) {
    alert("请选择时间变量");
    return;
}

// 获取 card-body 元素
const cardBody = document.getElementById(canvasId).closest('.card-body');

// 获取 timeLegend 元素
const timeLegend = document.getElementById('timeLegend');

// 移除之前的 canvas 或 div 元素
const existingCanvas = document.getElementById(canvasId);
if (existingCanvas) {
    existingCanvas.remove();
}

if (selectedKeys.size === 1) {
    // 一维直方图
    // 创建一个新的 canvas 元素
    const newCanvas = document.createElement('canvas');
    newCanvas.id = canvasId;
    newCanvas.style.width = '100%';
    newCanvas.style.height = '200px';

    // 将新的 canvas 元素插入到原来 canvas 的位置
    cardBody.insertBefore(newCanvas, timeLegend);

    renderChartJsHistogram(canvasId, data, selectedKeys, keyColors);

    // 恢复 timeLegend 到初始化样式
    timeLegend.style.position = '';
    timeLegend.style.bottom = '';
    timeLegend.style.width = '';
    timeLegend.style.backgroundColor = '';
    timeLegend.style.zIndex = '';
} else if (selectedKeys.size === 2) {
    // 二维直方图
    // 创建一个新的 div 元素
    const newDiv = document.createElement('div');
    newDiv.id = canvasId;
    newDiv.style.width = '100%';
    newDiv.style.height = '400px';

    // 将新的 div 元素插入到原来 canvas 的位置
    cardBody.insertBefore(newDiv, timeLegend);

    renderEchartsHeatmap(newDiv, data2d, selectedKeys, keyColors);

    // 恢复 timeLegend 到初始化样式
    timeLegend.style.position = '';
    timeLegend.style.bottom = '';
    timeLegend.style.width = '';
    timeLegend.style.backgroundColor = '';
    timeLegend.style.zIndex = '';
} else if (selectedKeys.size === 3) {
    // 三维直方图
    // 创建一个新的 div 元素
    const newDiv = document.createElement('div');
    newDiv.id = canvasId;
    newDiv.style.width = '100%';
    newDiv.style.height = '400px';

    // 将新的 div 元素插入到原来 canvas 的位置
    cardBody.insertBefore(newDiv, timeLegend);

    // 使用 Plotly 绘制三维直方图
    renderPlotlyScatter3d(newDiv, data3d, selectedKeys);

    // 设置 timeLegend 的样式
    timeLegend.style.clear = 'both'; // 确保 timeLegend 不与前面的元素在同一行
    timeLegend.style.marginTop = '10px'; // 添加一些顶部间距

    // 将 timeLegend 元素插入到新的 div 元素的下方
    cardBody.insertBefore(timeLegend, newDiv.nextSibling);
}
}


function renderChartJsHistogram(canvasId, data, selectedKeys, keyColors) {
const ctx = document.getElementById(canvasId).getContext('2d');

// 销毁之前的 Chart 实例（如果存在）
if (currentChartInstances[canvasId]) {
    if (currentChartInstances[canvasId].destroy) {
        currentChartInstances[canvasId].destroy(); // Chart.js 实例
    } else if (currentChartInstances[canvasId].dispose) {
        currentChartInstances[canvasId].dispose(); // ECharts 实例
    }
    delete currentChartInstances[canvasId];
}

const allData = [];
const allCategories = new Set();

for (let key of selectedKeys) {
    if (data[key] && Array.isArray(data[key])) {
        data[key].forEach(item => {
            if (item && item.category !== undefined && item.value !== undefined) {
                allData.push({ key, category: parseFloat(item.category), value: item.value }); // 保留小数
                allCategories.add(parseFloat(item.category)); // 保留小数
            } else {
                console.warn(`Invalid data item for key ${key}:`, item);
            }
        });
    } else {
        console.warn(`Invalid data for key ${key}:`, data[key]);
    }
}
    
const categories = Array.from(allCategories).sort((a, b) => a - b);
console.log("Categories:", categories);

if (selectedKeys.size === 0) {
    alert("请选择时间变量");
    return;
}

const key = Array.from(selectedKeys)[0];
const dataset = {
    label: key,
    data: categories.map(category => {
        const item = data[key].find(d => d.category === category);
        return item ? item.value : 0;
    }),
    backgroundColor: keyColors[key], // Use the assigned color
    borderColor: keyColors[key], // Use the assigned color
    borderWidth: 1
};

const chartData = {
    labels: categories,
    datasets: [dataset]
};

const chartOptions = {
    scales: {
        x: {
            title: {
                display: true,
                text: 'Category'
            }
        },
        y: {
            title: {
                display: true,
                text: 'Value'
            }
        }
    }
};

// 重置 canvas 高度和样式
const canvas = document.getElementById(canvasId);
if (!initialCanvasWidth || !initialCanvasHeight) {
    // 获取初始宽度和高度
    initialCanvasWidth = canvas.offsetWidth;
    initialCanvasHeight = canvas.offsetHeight;
}

canvas.width = initialCanvasWidth;
canvas.height = initialCanvasHeight;
canvas.style.width = `${initialCanvasWidth}px`;
canvas.style.height = `${initialCanvasHeight}px`;

currentChartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: chartData,
    options: chartOptions
});
}

function renderEchartsHeatmap(container, data2d, selectedKeys, keyColors) {
// 销毁之前的 Chart 实例（如果存在）
if (currentChartInstances[container.id]) {
    if (currentChartInstances[container.id].destroy) {
        currentChartInstances[container.id].destroy(); // Chart.js 实例
    } else if (currentChartInstances[container.id].dispose) {
        currentChartInstances[container.id].dispose(); // ECharts 实例
    }
    delete currentChartInstances[container.id];
}

const keys = Array.from(selectedKeys);
const xKey = keys[0];
const yKey = keys[1];
const key = `${xKey}_vs_${yKey}`;

if (!data2d[key]) {
    console.error(`No 2D data found for key ${key}`);
    return;
}

const xCategories = Object.keys(data2d[key]).map(Number).sort((a, b) => a - b);
const yCategories = Object.keys(data2d[key][xCategories[0]]).map(Number).sort((a, b) => a - b);
console.log("xCategories:", xCategories);
console.log("yCategories:", yCategories);

const seriesData = [];
xCategories.forEach((x, xIndex) => {
    yCategories.forEach((y, yIndex) => {
        seriesData.push([xIndex, yIndex, data2d[key][x][y] || 0]);
    });
});

// 获取 card-body 的高度和 timeLegend 的高度
const cardBody = container.closest('.card-body');
const timeLegend = document.getElementById('timeLegend');
const cardBodyHeight = cardBody.offsetHeight;
const cardBodyWidth = cardBody.offsetWidth;
const timeLegendHeight = timeLegend.offsetHeight;

// 计算合适的单元格大小
const maxCanvasHeight = 600; // 最大画布尺寸
const maxCanvasWidth = 350;
const fixedCellSize = 30; // 固定单元格大小
const cellSize = Math.min(fixedCellSize, maxCanvasHeight / yCategories.length, maxCanvasWidth / xCategories.length); // 根据最大类别数调整单元格大小

// 计算合适的高度
const availableHeight = cardBodyHeight - timeLegendHeight; // 减去图例高度后的可用高度
const calculatedHeight = cellSize * yCategories.length; // 根据单元格大小计算的高度
const height = Math.min(calculatedHeight, availableHeight); // 取计算高度和可用高度的最小值

// 计算合适的宽度
const availableWidth = cardBodyWidth;
const calculatedWidth = cellSize * xCategories.length; // 根据单元格大小计算的宽度
const width = Math.min(calculatedWidth, maxCanvasHeight); // 取计算宽度和最大宽度的最小值

// 动态设置 container 的高度和宽度
const tmp_w = 150;
const tmp_h = 150;
container.style.height = `${height+tmp_h}px`; // 确保 CSS 高度与实际高度一致
container.style.width = `${width+tmp_w}px`; // 确保 CSS 宽度与实际宽度一致


const chartOptions = {
    xAxis: {
        type: 'category',
        data: xCategories,
        name: yKey,

        min: 0,
        max: xCategories.length - 1,

        axisLabel: {
            interval: 0 // Show all labels
        },
        axisTick: {
            interval: 0 // Show all ticks
        },
        axisLine: {
            show: true
        },
        splitLine: {
            show: true,
            lineStyle: {
                color: '#000000' // 设置分隔线颜色为黑色
            }
        },
        gridIndex: 0
    },
    yAxis: {
        type: 'category',
        data: yCategories,
        name: xKey,

        min: 0,
        max: yCategories.length - 1,
        axisLabel: {
            interval: 0 // Show all labels
        },
        axisTick: {
            interval: 0 // Show all ticks
        },
        axisLine: {
            show: true
        },
        splitLine: {
            show: true,
            lineStyle: {
                color: '#000000' // 设置分隔线颜色为黑色
            }
        },
        gridIndex: 0
    },
    visualMap: {
        min: 0,
        max: Math.max(...seriesData.map(d => d[2])),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
            color: ['#F5FAEA', '#C3F94D', '#F7F94B', '#ffdd33', '#fdae61', '#d73027'] // 从绿色到浅绿，再到黄色、橙色、浅红，最后是深红色
        }
    },
    series: [{
        name: '2D Histogram',
        type: 'heatmap',
        data: seriesData
    }],
    grid: {
        left: '15%', // 增加左边距
        right: '5%', // 增加右边距
        top: '15%', // 增加上边距
        bottom: '10%', // 增加下边距
        width: `${width}px`, // 使用计算宽度
        height: `${height}px` // 使用计算高度
    }
};

const chartDom = container;
const myChart = echarts.init(chartDom);
myChart.setOption(chartOptions);
currentChartInstances[container.id] = myChart;
}

function renderPlotlyScatter3d(container, data3d, selectedKeys) {
// 销毁之前的 Plotly 实例（如果存在）
if (currentChartInstances[container.id]) {
    Plotly.purge(container);
    delete currentChartInstances[container.id];
}

const keys = Array.from(selectedKeys);
const xKey = keys[0];
const yKey = keys[1];
const zKey = keys[2];

// 尝试所有可能的特征顺序组合
const possibleKeys = [
    `${xKey}_vs_${yKey}_vs_${zKey}`,
    `${xKey}_vs_${zKey}_vs_${yKey}`,
    `${yKey}_vs_${xKey}_vs_${zKey}`,
    `${yKey}_vs_${zKey}_vs_${xKey}`,
    `${zKey}_vs_${xKey}_vs_${yKey}`,
    `${zKey}_vs_${yKey}_vs_${xKey}`
];

let foundKey = null;
for (const key of possibleKeys) {
    if (data3d[key]) {
        foundKey = key;
        break;
    }
}

if (!foundKey) {
    console.error(`No 3D data found for any key: ${possibleKeys.join(', ')}`);
    return;
}

const [xKeyFound, yKeyFound, zKeyFound] = foundKey.replace(/\(|\)/g, '').split('_vs_');

console.log('xKeyFound, yKeyFound, zKeyFound:', xKeyFound, yKeyFound, zKeyFound);

// 提取 xCategories 和 yCategories
const xCategories = Object.keys(data3d[foundKey]).map(key => {
    const parts = key.match(/\((\d+),\s*(\d+)\)/);
    if (parts && parts.length === 3) {
        return Number(parts[1]);
    }
    return NaN;
}).filter(value => !isNaN(value)).sort((a, b) => a - b);

const yCategories = Object.keys(data3d[foundKey]).map(key => {
    const parts = key.match(/\((\d+),\s*(\d+)\)/);
    if (parts && parts.length === 3) {
        return Number(parts[2]);
    }
    return NaN;
}).filter(value => !isNaN(value)).sort((a, b) => a - b);

console.log(`xCategories: ${xCategories}`);
console.log(`yCategories: ${yCategories}`);

// 确保 xCategories 和 yCategories 不为空
if (xCategories.length === 0 || yCategories.length === 0) {
    console.error(`xCategories or yCategories is empty for key ${foundKey}`);
    return;
}

// 提取 zCategories
const zCategories = Object.keys(data3d[foundKey][`(${xCategories[0]}, ${yCategories[0]})`]).map(Number).sort((a, b) => a - b);

// 确保 zCategories 不为空
if (zCategories.length === 0) {
    console.error(`zCategories is empty for key ${foundKey} with x=${xCategories[0]}, y=${yCategories[0]}`);
    return;
}

const dataPoints = [];
for (let x of xCategories) {
    for (let y of yCategories) {
        for (let z of zCategories) {
            const value = data3d[foundKey][`(${x}, ${y})`][z] || 0;
            dataPoints.push({ x: x, y: y, z: z, v: value });
        }
    }
}

const chartData = [{
    type: 'scatter3d',
    mode: 'markers',
    marker: {
        size: 12,
        line: {
            color: 'rgba(255, 255, 255, 0.8)',
            width: 0.5
        },
        opacity: 0.8,
        color: dataPoints.map(d => d.v), // Use the value as color
        colorscale: 'Viridis', // Define the colorscale
        colorbar: {
            title: 'Value'
        }
    },
    x: dataPoints.map(d => d.x),
    y: dataPoints.map(d => d.y),
    z: dataPoints.map(d => d.z),
    text: dataPoints.map(d => `x: ${d.x}, y: ${d.y}, z: ${d.z}, v: ${d.v}`), // 添加悬浮显示的文本
    hoverinfo: 'text' // 设置悬浮显示的信息为 text
}];

const chartOptions = {
    scene: {
        xaxis: {
            title: {
                text: zKeyFound
            }
        },
        yaxis: {
            title: {
                text: yKeyFound
            }
        },
        zaxis: {
            title: {
                text: xKeyFound
            }
        }
    },
    // Set willReadFrequently to true to optimize performance
    config: {
        responsive: true,
        toImageButtonOptions: {
            format: 'svg', // one of png, svg, jpeg, webp
            filename: 'custom_image',
            height: 500,
            width: 700,
            scale: 1 // Multiply title/legend/axis/canvas sizes by this factor
        },
        canvas: {
            willReadFrequently: true
        }
    }
};

// 使用 Plotly 绘制三维直方图
Plotly.newPlot(container, chartData, chartOptions);
currentChartInstances[container.id] = container;





// 获取 card-body 的高度
const cardBody = container.closest('.card-body');
const cardBodyHeight = cardBody.offsetHeight;

// 获取 timeLegend 的高度
const timeLegend = document.getElementById('timeLegend');
const timeLegendHeight = timeLegend.offsetHeight;

// 设置 timeLegend 的样式
timeLegend.style.position = 'absolute'; // 使用绝对定位
timeLegend.style.bottom = '200px'; // 设置距离底部的高度
timeLegend.style.width = '80%'; // 设置宽度为100%
timeLegend.style.backgroundColor = 'white'; // 设置背景颜色为白色
timeLegend.style.zIndex = '10'; // 设置 z-index 以确保在图表之上

// 设置 card-body 的相对定位
cardBody.style.position = 'relative';
}






function renderParallelCoordinates(data) {
console.log("Rendering parallel coordinates:", data);

// 获取 card-body 元素
const cardBody = document.getElementById('parallelCoordinatesContainer').closest('.card-body');

// 移除之前的 div 元素
const existingDiv = document.getElementById('parallelCoordinatesContainer');
if (existingDiv) {
    existingDiv.remove();
}

// 创建一个新的 div 元素
const newDiv = document.createElement('div');
newDiv.id = 'parallelCoordinatesContainer';
newDiv.style.width = '100%';
newDiv.style.height = '600px'; // 修改高度为600px

// 将新的 div 元素插入到原来 div 的位置
cardBody.insertBefore(newDiv, null);

// 处理数据
const dimensions = [];
const seriesData = [];

// 提取目标变量和其他特征的键
const targetKeys = Object.keys(data[0].target_variable);
const otherKeys = Object.keys(data[0].other_feature);

// 添加目标变量的维度，并计算 min 和 max
targetKeys.forEach((key, index) => {
    const values = data.map(item => item.target_variable[key]);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    dimensions.push({
        name: key,
        type: 'value',
        dim: index, // 添加 dim 属性
        min: minValue, // 设置 y 轴从最小值开始
        max: maxValue // 设置 y 轴到最大值结束
    });
});

// 添加其他特征的维度，并计算 min 和 max
otherKeys.forEach((key, index) => {
    const values = data.map(item => item.other_feature[key]);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    dimensions.push({
        name: key,
        type: 'value',
        dim: targetKeys.length + index, // 添加 dim 属性
        min: minValue, // 设置 y 轴从最小值开始
        max: maxValue // 设置 y 轴到最大值结束
    });
});

// 构建 seriesData
data.forEach(item => {
    const targetValues = Object.values(item.target_variable);
    const otherValues = Object.values(item.other_feature);
    seriesData.push([...targetValues, ...otherValues]);
});

console.log("Dimensions:", dimensions);
console.log("Series Data:", seriesData);

// 初始化 ECharts 实例
const chartDom = document.getElementById('parallelCoordinatesContainer');
const myChart = echarts.init(chartDom);

console.log("ECharts Instance:", myChart); // 添加日志

// 配置项
const option = {
    parallelAxis: dimensions,
    tooltip: {
        trigger: 'item',
        axisPointer: {
            type: 'line'
        },
        position: function (point, params, dom, rect, size) {
            // 自定义 tooltip 位置
            return [point[0] + 10, point[1] - 20];
        },
        formatter: function (params) {
            let tooltip = '';
            params.value.forEach((value, index) => {
                tooltip += `${dimensions[index].name}: ${value}<br>`;
            });
            return tooltip;
        }
    },
    series: [
        {
            type: 'parallel',
            data: seriesData,
            lineStyle: {
                width: 1,
                opacity: 0.5,
                emphasis: {
                    width: 3,
                    opacity: 1,
                    color: 'red' // 设置高亮时的线段颜色为红色
                }
            },
            smooth: true,
            emphasis: {
                lineStyle: {
                    width: 3,
                    opacity: 1,
                    color: 'red' // 设置高亮时的线段颜色为红色
                }
            }
        }
    ]
};

console.log("Option:", option); // 添加日志

myChart.setOption(option);

// 添加鼠标悬浮事件来高亮折线
myChart.on('mouseover', function (params) {
    if (params.componentType === 'series' && params.seriesType === 'parallel') {
        const dataIndex = params.dataIndex;
        myChart.dispatchAction({
            type: 'downplay',
            seriesIndex: 0
        });
        myChart.dispatchAction({
            type: 'highlight',
            seriesIndex: 0,
            dataIndex: dataIndex
        });
    }
});

myChart.on('mouseout', function (params) {
    if (params.componentType === 'series' && params.seriesType === 'parallel') {
        myChart.dispatchAction({
            type: 'downplay',
            seriesIndex: 0
        });
    }
});
}
