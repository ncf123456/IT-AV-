// Function to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

let trainingInterval; // Define trainingInterval here
let timeData = {};
let timeData2d = {};
let timeData3d = {};
let geoData = {};
let geoData2d = {};
let geoData3d = {};

const defaultColors = [
    '#5470c6','#91cc75', '#fac858', '#ee6666', '#73c0de','#3ba272'
];

const lightColors = [
    '#c2d4f0', '#d7e8bf', '#ffe5b3', '#ffcdd2', '#cfeef9', '#a6d9b1'
];

const previousColors = ['#fc8452', '#9a60b4','#ea7ccc', '#546570', '#c4ccd3']
const previousLightColors=['#ffccab', '#d3bbdf', '#ffd5dc', '#b3c7d6', '#e5e9f0']

const colorPalette = [
    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40',
    '#E7E9ED', '#D35400', '#1ABC9C', '#2ECC71', '#3498DB', '#9B59B6',
    '#34495E', '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#2C3E50',
    '#F39C12', '#E67E22'
];

document.addEventListener('DOMContentLoaded', () => {
    // 自动显示帮助模态框
    $('#helpModal').modal('show');

    // 初始化并绘制流程图
    initFlowChart();

    const projectId = getProjectId(); // Ensure this function returns the correct project ID
    const socket = new WebSocket(`ws://${window.location.host}/ws/training_progress/${projectId}/`);

    socket.onopen = function(event) {
        console.log("Connected to WebSocket");
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const progress = data.progress;
        const progressBar = document.getElementById('trainingProgress');
        if (progressBar) {
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';
        }
        if (progress >= 100) {
            clearInterval(trainingInterval); // Ensure trainingInterval is defined
            document.getElementById('saveButton').disabled = false;
            console.log("100%达到");

            
        }

        
    };

    socket.onclose = function(event) {
        console.log("Disconnected from WebSocket");
    };

    // 添加页面卸载事件监听器
    window.addEventListener('beforeunload', function() {
            deletePreviousFiles();
        });

    // 添加事件监听器来更新滑动条的数值显示
    const rangeInputs = document.querySelectorAll('.form-control-range');
    rangeInputs.forEach(input => {
        input.addEventListener('input', function() {
            const valueDisplay = document.getElementById(this.id + '_value');
            valueDisplay.textContent = this.value;
        });
    });

    const bayesianOptimizationCallsInput = document.getElementById('bayesianOptimizationCalls');
    const bayesianOptimizationCallsValueSpan = document.getElementById('bayesianOptimizationCallsValue');

    // 为 bayesianOptimizationCalls 滑块添加 input 事件监听器
    bayesianOptimizationCallsInput.addEventListener('input', function() {
        bayesianOptimizationCallsValueSpan.textContent = this.value;
    });

    
    
});


function initFlowChart() {
    const $ = go.GraphObject.make;

    // 创建三个 Diagram 实例，分别对应三个流程图
    const myDiagram1 = $(go.Diagram, "myDiagramDiv1",
        {
            "undoManager.isEnabled": true,
            "layout": $(go.LayeredDigraphLayout, { direction: 90, layerSpacing: 10 }), // 缩小上下节点之间的距离
            "initialContentAlignment": go.Spot.Center,// 使流程图居中    
        });

    const myDiagram2 = $(go.Diagram, "myDiagramDiv2",
        {
            "undoManager.isEnabled": true,
            "layout": $(go.LayeredDigraphLayout, { direction: 90, layerSpacing: 10 }), // 缩小上下节点之间的距离
            "initialContentAlignment": go.Spot.Center // 使流程图居中
        });

    const myDiagram3 = $(go.Diagram, "myDiagramDiv3",
        {
            "undoManager.isEnabled": true,
            "layout": $(go.LayeredDigraphLayout, { direction: 90, layerSpacing: 10 }), // 缩小上下节点之间的距离
            "initialContentAlignment": go.Spot.Center // 使流程图居中
        });

    // 定义节点模板
    const nodeTemplate = $(go.Node, "Auto",
        $(go.Shape, "RoundedRectangle", { fill: "lightblue" },
            new go.Binding("fill", "color")),
        $(go.TextBlock,
            { margin: 8 , stroke: "white"},  // some room around the text
            new go.Binding("text", "key"))
    );

    // 定义链接模板
    const linkTemplate = $(go.Link, go.Link.Orthogonal,
        $(go.Shape, { stroke: "black" }),
        $(go.Shape, { toArrow: "Standard", stroke: null, fill: "black" })
    );

    // 定义流程一的节点和链接
    // Define nodes and links for process one
    const nodeDataArray1 = [
        { key: "Select Training File", color: "#51B1E1" }, 
        { key: "Select Feature Columns", color: "#51B1E1" }, 
        { key: "View Data", color: "#F95F53" }, 
        { key: "Train", color: "#1F3BB3" }, 
        { key: "Adjust Parameters Based on Suggestions or Manually", color: "#51B1E1" }, 
        { key: "Save", color: "#34B1AA" } 
    ];

    const linkDataArray1 = [
        { from: "Select Training File", to: "Select Feature Columns" },
        { from: "Select Feature Columns", to: "View Data" },
        { from: "View Data", to: "Train" },
        { from: "Train", to: "Adjust Parameters Based on Suggestions or Manually" },
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Train", key: "loop1" }, // Loop arrow 1
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Save" }
    ];

    // Define nodes and links for process two
    const nodeDataArray2 = [
        { key: "Select Training File", color: "#51B1E1" }, 
        { key: "Select Feature Columns", color: "#51B1E1" }, 
        { key: "Narrow Parameter Range", color: "#E29E09" }, 
        { key: "Train", color: "#1F3BB3" }, 
        { key: "Adjust Parameters Based on Suggestions or Manually", color: "#51B1E1" }, 
        { key: "Save", color: "#34B1AA" } 
    ];

    const linkDataArray2 = [
        { from: "Select Training File", to: "Select Feature Columns" },
        { from: "Select Feature Columns", to: "Narrow Parameter Range" },
        { from: "Narrow Parameter Range", to: "Train" },
        { from: "Train", to: "Adjust Parameters Based on Suggestions or Manually" },
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Train", key: "loop1" }, // Loop arrow 1
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Save" }
    ];

    // Define nodes and links for process three
    const nodeDataArray3 = [
        { key: "Select Training File", color: "#51B1E1" }, 
        { key: "Select Feature Columns", color: "#51B1E1" }, 
        { key: "Parameter Selection", color: "#51B1E1" }, 
        { key: "Train", color: "#1F3BB3" }, 
        { key: "Adjust Parameters Based on Suggestions or Manually", color: "#51B1E1" }, 
        { key: "Save", color: "#34B1AA" } 
    ];

    const linkDataArray3 = [
        { from: "Select Training File", to: "Select Feature Columns" },
        { from: "Select Feature Columns", to: "Parameter Selection" },
        { from: "Parameter Selection", to: "Train" },
        { from: "Train", to: "Adjust Parameters Based on Suggestions or Manually" },
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Train", key: "loop1" }, // Loop arrow 1
        { from: "Adjust Parameters Based on Suggestions or Manually", to: "Save" }
    ];

    // 设置每个 Diagram 的模板和数据
    myDiagram1.nodeTemplate = nodeTemplate;
    myDiagram1.linkTemplate = linkTemplate;
    myDiagram1.model = new go.GraphLinksModel(nodeDataArray1, linkDataArray1);

    myDiagram2.nodeTemplate = nodeTemplate;
    myDiagram2.linkTemplate = linkTemplate;
    myDiagram2.model = new go.GraphLinksModel(nodeDataArray2, linkDataArray2);

    myDiagram3.nodeTemplate = nodeTemplate;
    myDiagram3.linkTemplate = linkTemplate;
    myDiagram3.model = new go.GraphLinksModel(nodeDataArray3, linkDataArray3);
}



function deletePreviousFiles() {
        const projectId = getProjectId();
        const urls = [
            `/data/delete_previous_file/${projectId}/metrics/`,
            `/data/delete_previous_file/${projectId}/loss/`,
            `/data/delete_previous_file/${projectId}/predictions/`
        ];

        urls.forEach(url => {
            fetch(url, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken,
                },
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Data from server deletePreviousFiles:", data);
            })
            .catch(error => {
                console.error('Error deleting previous files:', error);
            });
        });
    }


function startTraining() {
    console.log("Starting training...");
    const progressBar = document.getElementById('trainingProgress');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';

    const formData = new FormData(document.getElementById('trainingForm'));
    const projectId = getProjectId();

    fetch(`/data/data_training_dash/${projectId}/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
        },
        body: formData
    })
    .then(response => {
        console.log("Response startTraining received:", response);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Data from server startTraining:", data);
        if (data.status === 'success') {
            clearInterval(trainingInterval);
            document.getElementById('saveButton').disabled = false;
            console.log("Training completed.");
            fetchDataAndRenderCharts(projectId, formData.get('data_file'));
            if (data.triggered_epoch !== -1) {  // 检查早停触发的轮次
                alert(`早停在第 ${data.triggered_epoch} 轮触发`);
            }
            fetchSuggestionsAndShowModal(projectId); // 新增：获取建议并显示模态窗口
        } else {
            showErrorMessage(data.message);
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        showErrorMessage('发生错误，请检查控制台以获取更多信息。');
    });
}
    



function fetchSuggestionsAndShowModal(projectId) {
    console.log("Fetching suggestions...");
    fetch(`/data/get_suggestions/${projectId}/`)
        .then(response => {
            console.log("Response fetchSuggestionsAndShowModal received:", response);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Data from server fetchSuggestionsAndShowModal:", data);
            if (data.status === 'success') {
                console.log("Suggestions received:", data.suggestions);
                // 打开新窗口并传递 responseData
                // const newWindow = window.open('', '_blank', 'width=600,height=400');
                // newWindow.document.write(`
                //     <html>
                //     <head>
                //         <title>参数建议</title>
                //         <style>
                //             .param-section {
                //                 margin-left: 20px; /* 整体向右偏移一点 */
                //             }

                //             .param-section > label,
                //             .param-section > h5,
                //             .param-section > span {
                //                 margin-left: 20px; /* 在 .param-section 中的 label, h5, span 等元素再往右偏移一些 */
                //             }
                            
                //             .param-section > * {
                //                 margin-bottom: 5px; /* 减小 .param-section 内部元素的上下间距 */
                //             }
                //             .label-large {
                //                 font-size: 1.2em; /* 增加 label 的字体大小 */
                //                 font-weight: bold; /* 加粗 label 的字体 */
                //             }
                //             .modal-title {
                //                 margin-top: 20px; /* 增加 .modal-title 与上方元素的间距 */
                //             }
                            
                //             /* 新增样式以减小 h5 和 h5 之间的间距 */
                //             .param-section h5 + h5 {
                //                 margin-top: 5px;
                //             }
                        
                //             /* 新增样式以增大 label 和 h5 之间的间距 */
                //             .param-section > label + h5 {
                //                 margin-top: 15px;
                //             }
                        
                //             /* 新增样式以增大 .param-section 和 .param-section 之间的距离 */
                //             .param-section + .param-section {
                //                 margin-top: 40px;
                //             }
                        
                //             /* 新增样式以增大 .param-section 和 .modal-title 之间的距离 */
                //             .param-section + h3.modal-title {
                //                 margin-top: 60px;
                //             }
                //         </style>
                //     </head>
                //     <body>
                //         <div id="recommendationModalBody">
                //             ${data.suggestions.map(suggestion => `
                //                 <h3 class="modal-title">目标变量: ${suggestion.target}</h3>
                //                 <div class="param-section">
                //                     <label class="label-large">Temporal Kernel Size：</label>
                //                     <h5>${suggestion.suggestions.temporal_kernel_size}</h5>
                //                     <h5>${suggestion.suggestions.temporal_kernel_size_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">LSTM Units：</label>
                //                     <h5>${suggestion.suggestions.lstm_units}</h5>
                //                     <h5>${suggestion.suggestions.lstm_units_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Spatial Kernel Sizes：</label>
                //                     <h5>${suggestion.suggestions.spatial_kernel_sizes}</h5>
                //                     <h5>${suggestion.suggestions.spatial_kernel_sizes_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Filters：</label>
                //                     <h5>${suggestion.suggestions.filters}</h5>
                //                     <h5>${suggestion.suggestions.filters_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Dynamic Attention Units：</label>
                //                     <h5>${suggestion.suggestions.dynamic_attention_units}</h5>
                //                     <h5>${suggestion.suggestions.dynamic_attention_units_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Dense Units：</label>
                //                     <h5>${suggestion.suggestions.dense_units}</h5>
                //                     <h5>${suggestion.suggestions.dense_units_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Epochs：</label>
                //                     <h5>${suggestion.suggestions.epochs}</h5>
                //                     <h5>${suggestion.suggestions.epochs_reason}</h5>
                //                 </div>
                //                 <div class="param-section">
                //                     <label class="label-large">Batch Size：</label>
                //                     <h5>${suggestion.suggestions.batch_size}</h5>
                //                     <h5>${suggestion.suggestions.batch_size_reason}</h5>
                //                 </div>
                //             `).join('\n')}
                //         </div>
                //     </body>
                //     </html>
                // `);
            } else {
                showErrorMessage(data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching suggestions:', error);
            showErrorMessage('');
        });
}
    




document.getElementById('dataFileSelect').addEventListener('change', function() {
    const selectedFile = this.value;
    if (selectedFile) {
        const projectId = getProjectId();
        fetch(`/data/get_columns/${projectId}/${encodeURIComponent(selectedFile)}/`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                populateFeatureColumns(data.columns);
                document.getElementById('trainButton').disabled = false; // Enable train button after successful fetch
                document.getElementById('viewDataButton').disabled = false;
                document.getElementById('bayesianOptimizationButton').disabled = false;
                clearErrorMessage(); // Clear any previous error messages
                fetch(`/data/get_data_size/${projectId}/${encodeURIComponent(selectedFile)}/`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        const dataSize = data.data_size;
                        console.log("Data size:", dataSize);
                        // setRangeLimits(dataSize);
                    })
                    .catch(error => {
                        console.error('Error fetching data size:', error);
                    });
            })
            .catch(error => {
                console.error('Error fetching columns:', error);
                showErrorMessage('无法获取列信息，请检查控制台以获取更多信息。');
                document.getElementById('trainButton').disabled = true; // Disable train button on failure
                hideAllFeatureSections(); // Hide all feature sections on failure
            });
    } else {
        hideAllFeatureSections();
        document.getElementById('trainButton').disabled = true; // Disable train button when no file is selected
        clearErrorMessage(); // Clear any previous error messages
    }
});

function setRangeLimits(dataSize) {
    let epochsMin, epochsMax, batchSizeMin, batchSizeMax;

    if (dataSize < 500) {
        epochsMin = 50;
        epochsMax = 200;
        batchSizeMin = 8;
        batchSizeMax = 64;
    }else if(dataSize < 1000){
        epochsMin = 30;
        epochsMax = 100;
        batchSizeMin = 16;
        batchSizeMax = 128;
    }
    else if(dataSize < 3000){
        epochsMin = 20;
        epochsMax = 80;
        batchSizeMin = 16;
        batchSizeMax = 128;
    }
     else if (dataSize < 5000) {
        epochsMin = 10;
        epochsMax = 70;
        batchSizeMin = 16;
        batchSizeMax = 128;
    } else {
        epochsMin = 5;
        epochsMax = 50;
        batchSizeMin = 32;
        batchSizeMax = 256;
    }


    const epochsInput = document.getElementById('epochs');
    epochsInput.min = epochsMin;
    epochsInput.max = epochsMax;
    epochsInput.value = epochsMin;
    document.getElementById('epochs_value').textContent = epochsMin;

    const batchSizeInput = document.getElementById('batch_size');
    batchSizeInput.min = batchSizeMin;
    batchSizeInput.max = batchSizeMax;
    batchSizeInput.value = batchSizeMin;
    document.getElementById('batch_size_value').textContent = batchSizeMin;

    // 更新 epochs 的 min 和 max 显示
    document.getElementById('epochs-min').textContent = epochsMin;
    document.getElementById('epochs-max').textContent = epochsMax;

    // 更新 batch_size 的 min 和 max 显示
    document.getElementById('batch_size-min').textContent = batchSizeMin;
    document.getElementById('batch_size-max').textContent = batchSizeMax;
}


function populateFeatureColumns(columns) {
    const featureColumnsDiv = document.getElementById('featureColumns');
    featureColumnsDiv.innerHTML = '<h5>特征列选择</h5>';
    let rowDiv = document.createElement('div'); // 创建一个新的行 div
    rowDiv.className = 'row'; // 使用 Bootstrap 的 row 类

    columns.forEach((column, index) => {
        const columnDiv = document.createElement('div');
        columnDiv.className = 'col-sm-6 form-group d-flex align-items-center'; // 使用 col-sm-6 类使每两个元素在同一行

        columnDiv.innerHTML = `
            <label for="${column}" class="mr-2" style="margin-right: 10px;">${column}</label>
            <select class="form-control form-control-sm" 
                    style="width: 120px; height: 8px; font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px;" 
                    id="${column}" name="features[${column}]">
                
                <option value="other_feature">Other Feature</option>
                <option value="time_feature">Time Feature</option>
                <option value="standard_date_feature">Standard Date Feature</option>
                <option value="geographic_feature">Geographic Feature</option>
                <option value="target_variable">Target Variable</option>
            </select>
        `;

        rowDiv.appendChild(columnDiv); // 将列 div 添加到行 div 中

        // 每两个元素创建一个新的行 div
        if ((index + 1) % 2 === 0) {
            featureColumnsDiv.appendChild(rowDiv);
            rowDiv = document.createElement('div');
            rowDiv.className = 'row';
        }
    });

    // 如果最后一个行 div 中有元素，添加到 featureColumnsDiv 中
    if (rowDiv.children.length > 0) {
        featureColumnsDiv.appendChild(rowDiv);
    }

    featureColumnsDiv.style.display = '';
}

function hideAllFeatureSections() {
    document.getElementById('featureColumns').style.display = 'none';
}

function saveModel() {
    console.log("Saving model...");
    const projectId = getProjectId();
    const formData = new FormData(document.getElementById('trainingForm'));

    // 弹出输入框让用户输入模型名称
    const modelName = prompt("请输入模型名称:");
    if (modelName) {
        formData.append('model_name', modelName);

        fetch(`/data/save_model/${projectId}/`, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            body: formData,
        })
        .then(response => {
            console.log("Response saveModel received:", response);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Data from server saveModel:", data);
            if (data.status === 'success') {
                alert(data.message);
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error saving model:', error);
            alert('发生错误，请检查控制台以获取更多信息。');
        });
    } else {
        alert('模型名称不能为空');
    }
}

function getProjectId() {
    return document.getElementById('projectIdInput').value;
}



function generateAndFetchData() {
    console.log("Generating and fetching data for histograms and parallel coordinates...");
    const formData = new FormData(document.getElementById('trainingForm'));
    const projectId = getProjectId();

    fetch(`/data/generate_histograms_and_parallel_coordinates/${projectId}/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
        },
        body: formData
    })
    .then(response => {
        console.log("Response generateAndFetchData received:", response);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Data from server generateAndFetchData:", data);
        
        if (data.status === 'success') {
            fetchDataAndViewCharts(projectId, formData.get('data_file'));
        } else {
            showErrorMessage(data.message);
        }
        if (data.status === 'success') {
            // 打印 JSON 字符串以调试
            console.log("JSON.stringify(data):", JSON.stringify(data));
        
            // 打开新窗口并传递 responseData
    //         const newWindow = window.open('', '_blank', 'width=600,height=400');
    //         newWindow.document.write(`
    //             <html>
    //             <head>
    //                 <title>推荐参数</title>
    //                 <style>

    //                     .param-section {
    //                             margin-left: 20px; /* 整体向右偏移一点 */
    //                         }

    //                     .param-section > label,
    //                     .param-section > h5,
    //                     .param-section > span {
    //                         margin-left: 20px; /* 在 .param-section 中的 label, h5, span 等元素再往右偏移一些 */
    //                     }
                        
    //                     .param-section > * {
    //                         margin-bottom: 5px; /* 减小 .param-section 内部元素的上下间距 */
    //                     }
    //                     .label-large {
    //                         font-size: 1.2em; /* 增加 label 的字体大小 */
    //                         font-weight: bold; /* 加粗 label 的字体 */
    //                     }
    //                     .modal-title {
    //                         margin-top: 20px; /* 增加 .modal-title 与上方元素的间距 */
    //                     }
                        
    //                     /* 新增样式以减小 h5 和 h5 之间的间距 */
    //                     .param-section h5 + h5 {
    //                         margin-top: 5px;
    //                     }
                    
    //                     /* 新增样式以增大 label 和 h5 之间的间距 */
    //                     .param-section > label + h5 {
    //                         margin-top: 15px;
    //                     }
                    
    //                     /* 新增样式以增大 .param-section 和 .param-section 之间的距离 */
    //                     .param-section + .param-section {
    //                         margin-top: 40px;
    //                     }
                    
    //                     /* 新增样式以增大 .param-section 和 .modal-title 之间的距离 */
    //                     .param-section + h3.modal-title {
    //                         margin-top: 60px;
    //                     }
    //                 </style>
    //                 <script>
    //                     // 定义 agreeAndShrinkButtonHandler 函数
    //                     function agreeAndShrinkButtonHandler() {
    //                         const responseData = JSON.parse(document.getElementById('responseDataInput').value);
    //                         const paramIds = [
    //                             'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
    //                             'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
    //                         ];
        
    //                         const recommendations = [
    //                             responseData.temporal_kernel_size_recommendation,
    //                             responseData.lstm_units_recommendation,
    //                             responseData.spatial_kernel_sizes_recommendation,
    //                             responseData.filters_recommendation,
    //                             responseData.dynamic_attention_units_recommendation,
    //                             responseData.dense_units_recommendation,
    //                             responseData.epochs_recommendation,
    //                             responseData.batch_size_recommendation
    //                         ];
        
    //                         recommendations.forEach((recommendation, index) => {
    //                             // 确保 recommendation 是数组
    //                             if (!Array.isArray(recommendation)) {
    //                                 recommendation = [recommendation];
    //                             }
        
    //                             const input = window.opener.document.getElementById(paramIds[index]);
    //                             const min = Math.min(...recommendation);
    //                             const max = Math.max(...recommendation);
    //                             const value = Math.floor((min + max) / 2);
        
    //                             input.min = min;
    //                             input.max = max;
    //                             input.value = value;
        
    //                             const minSpan = window.opener.document.querySelector(\`label[for="\${paramIds[index]}"] .range-min\`);
    //                             const maxSpan = window.opener.document.querySelector(\`label[for="\${paramIds[index]}"] .range-max\`);
        
    //                             if (minSpan && maxSpan) {
    //                                 minSpan.textContent = min;
    //                                 maxSpan.textContent = max;
    //                             }
        
    //                             const valueDisplay = window.opener.document.getElementById(paramIds[index] + '_value');
    //                             if (valueDisplay) {
    //                                 valueDisplay.textContent = value;
    //                             }
    //                         });
        
    //                         window.close();
    //                     }
        
    //                     // 定义 shrinkValuesButtonHandler 函数
    //                     function shrinkValuesButtonHandler() {
    //                         const responseData = JSON.parse(document.getElementById('responseDataInput').value);
    //                         const paramIds = [
    //                             'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
    //                             'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
    //                         ];
        
    //                         const recommendations = [
    //                             responseData.temporal_kernel_size_recommendation,
    //                             responseData.lstm_units_recommendation,
    //                             responseData.spatial_kernel_sizes_recommendation,
    //                             responseData.filters_recommendation,
    //                             responseData.dynamic_attention_units_recommendation,
    //                             responseData.dense_units_recommendation,
    //                             responseData.epochs_recommendation,
    //                             responseData.batch_size_recommendation
    //                         ];
        
    //                         recommendations.forEach((recommendation, index) => {
    //                             const input = window.opener.document.getElementById(paramIds[index]);
    //                             const min = Math.min(...recommendation);
    //                             const max = Math.max(...recommendation);
    //                             const value = Math.floor((min + max) / 2);
        
                                
    //                             input.value = value;
        
    //                             const valueDisplay = window.opener.document.getElementById(paramIds[index] + '_value');
    //                             if (valueDisplay) {
    //                                 valueDisplay.textContent = value;
    //                             }
    //                         });
        
    //                         window.close();
    //                     }
        
    //                     // 定义 cancelButtonHandler 函数
    //                     function cancelButtonHandler() {
    //                         window.close();
    //                     }
    //                 </script>
    //             </head>
    //             <body>
    //     <div id="recommendationModalBody">
    //         <h3 class="modal-title">时空感知相关参数：</h3>
    //         <div class="param-section">
    //             <label class="label-large">Temporal Kernel Size：</label>
    //             <h5>${data.temporal_kernel_size_recommendation_message}</h5>
    //             <h5>${data.temporal_kernel_size_recommendation_details}</h5>
    //             ${data.fft_detail_messages.map(msg => `<br><span class="d-block">${msg}</span>`).join('\n')}
    //         </div>
    //         <div class="param-section">
    //             <label class="label-large">LSTM Units：</label>
    //             <h5>${data.lstm_units_recommendation_message}</h5>
    //             <h5>${data.lstm_units_recommendation_details}</h5>
    //             ${data.acf_detail_messages.map(msg => `<br><span class="d-block">${msg}</span>`).join('\n')}
    //         </div>
    //         <div class="param-section">
    //             <label class="label-large">Spatial Kernel Sizes：</label>
    //             <h5>${data.spatial_kernel_sizes_recommendation_message}</h5>
    //             <h5>${data.spatial_kernel_sizes_recommendation_details}</h5>
    //             <span class="d-block">${data.num_clusters_message}</span>
    //             ${data.cluster_counts_messages.map(msg => `<br><span class="d-block" style="margin-left: 24px;">${msg}</span>`).join('\n')}
    //         </div>
    //         <div class="param-section">
    //             <label class="label-large">Filters：</label>
    //             <h5>${data.filters_recommendation_message}</h5>
    //             <h5>${data.filters_recommendation_details}</h5>
    //             ${data.std_dev_detail_messages.map(msg => `<br><span class="d-block" style="margin-left: 24px;">${msg}</span>`).join('\n')}
    //         </div>
    //         <h3 class="modal-title">注意力机制和全连接层：</h3>
    //         <div class="param-section">
    //             <label class="label-large">Dynamic Attention Units：</label>
    //             <h5>${data.dynamic_attention_units_recommendation_message}</h5>
    //             <h5>${data.dynamic_attention_units_recommendation_details}</h5>
    //             ${data.importance_detail_messages.map(msg => `<br><span class="d-block" style="margin-left: 24px;">${msg}</span>`).join('\n')}
    //         </div>
    //         <div class="param-section">
    //             <label class="label-large">Dense Units：</label>
    //             <h5>${data.dense_units_recommendation_message}</h5>
    //             <h5>${data.dense_units_recommendation_details}</h5>
    //             ${data.corr_detail_messages.map(msg => `<br><span class="d-block" style="margin-left: 24px;">${msg}</span>`).join('\n')}
    //         </div>
    //         <h3 class="modal-title">训练相关参数：</h3>
    //         <div class="param-section">
    //             <label class="label-large">Epochs：</label>
    //             <h5>${data.data_size_message}</h5>
    //             <h5>${data.epochs_recommendation_message}</h5>
    //             <h5>${data.epochs_recommendation_details}</h5>
    //         </div>
    //         <div class="param-section">
    //             <label class="label-large">Batch Size：</label>
    //             <h5>${data.data_size_message}</h5>
    //             <h5>${data.batch_size_recommendation_message}</h5>
    //             <h5>${data.batch_size_recommendation_details}</h5>
    //         </div>
    //     </div>
    //     <input type="hidden" id="responseDataInput" value='${JSON.stringify(data)}'>
    //     <button onclick="agreeAndShrinkButtonHandler()">同意推荐，缩小参数范围</button>
    //     <button onclick="shrinkValuesButtonHandler()">不缩小范围，仅更新推荐参数值</button>
    //     <button onclick="cancelButtonHandler()">取消</button>
    // </body>
    //             </html>
    //         `);
        } else {
            showErrorMessage(data.message);
        }
    })
    .catch(error => {
        console.error('Error generating and fetching data:', error);
        showErrorMessage('发生错误，请检查控制台以获取更多信息。');
    });
}

function showRecommendationModal(responseData) {
    const modalBody = document.getElementById('recommendationModalBody');
    modalBody.innerHTML = '';

    const paramNames = [
        'Temporal Kernel Size', 'LSTM Units', 'Spatial Kernel Size', 
        'Filters', 'Dynamic Attention Units', 'Dense Units', 'Epochs', 'Batch Size'
    ];

    const paramIds = [
        'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
        'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
    ];

    const recommendations = [
        responseData.temporal_kernel_size_recommendation,
        responseData.lstm_units_recommendation,
        responseData.spatial_kernel_sizes_recommendation,
        responseData.filters_recommendation,
        responseData.dynamic_attention_units_recommendation,
        responseData.dense_units_recommendation,
        responseData.epochs_recommendation,
        responseData.batch_size_recommendation
    ];

    recommendations.forEach((recommendation, index) => {
        // 确保 recommendation 是数组
        if (!Array.isArray(recommendation)) {
            recommendation = [recommendation];
        }

        const div = document.createElement('div');
        div.className = 'form-group d-flex align-items-center';
        div.innerHTML = `
            <label class="mr-2 small-label">${paramNames[index]}: ${recommendation.join('-')}</label>
        `;
        modalBody.appendChild(div);
    });

    // 将 responseData 传递给按钮事件处理函数
    document.getElementById('agreeAndShrinkButton').addEventListener('click', () => agreeAndShrinkButtonHandler(responseData));
    document.getElementById('shrinkValuesButton').addEventListener('click', () => shrinkValuesButtonHandler(responseData));

    $('#recommendationModal').modal('show');
}

function agreeAndShrinkButtonHandler(responseData) {
    if (window.opener) {
        const paramIds = [
            'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
            'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
        ];

        const recommendations = [
            responseData.temporal_kernel_size_recommendation,
            responseData.lstm_units_recommendation,
            responseData.spatial_kernel_sizes_recommendation,
            responseData.filters_recommendation,
            responseData.dynamic_attention_units_recommendation,
            responseData.dense_units_recommendation,
            responseData.epochs_recommendation,
            responseData.batch_size_recommendation
        ];

        recommendations.forEach((recommendation, index) => {
            // 确保 recommendation 是数组
            if (!Array.isArray(recommendation)) {
                recommendation = [recommendation];
            }

            const input = window.opener.document.getElementById(paramIds[index]);
            const min = Math.min(...recommendation);
            const max = Math.max(...recommendation);
            const value = Math.floor((min + max) / 2);

            input.min = min;
            input.max = max;
            input.value = value;

            const minSpan = window.opener.document.querySelector(`label[for="${paramIds[index]}"] .range-min`);
            const maxSpan = window.opener.document.querySelector(`label[for="${paramIds[index]}"] .range-max`);

            if (minSpan && maxSpan) {
                minSpan.textContent = min;
                maxSpan.textContent = max;
            }

            const valueDisplay = window.opener.document.getElementById(paramIds[index] + '_value');
            if (valueDisplay) {
                valueDisplay.textContent = value;
            }
        });

        window.close();
    } else {
        console.error('window.opener is not available');
    }
}

function shrinkValuesButtonHandler(responseData) {
    if (window.opener) {
        const paramIds = [
            'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
            'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
        ];

        const recommendations = [
            responseData.temporal_kernel_size_recommendation,
            responseData.lstm_units_recommendation,
            responseData.spatial_kernel_sizes_recommendation,
            responseData.filters_recommendation,
            responseData.dynamic_attention_units_recommendation,
            responseData.dense_units_recommendation,
            responseData.epochs_recommendation,
            responseData.batch_size_recommendation
        ];

        recommendations.forEach((recommendation, index) => {
            const input = window.opener.document.getElementById(paramIds[index]);
            const min = Math.min(...recommendation);
            const max = Math.max(...recommendation);
            const value = Math.floor((min + max) / 2);

            input.min = min;
            input.max = max;
            input.value = value;

            const valueDisplay = window.opener.document.getElementById(paramIds[index] + '_value');
            if (valueDisplay) {
                valueDisplay.textContent = value;
            }
        });

        window.close();
    } else {
        console.error('window.opener is not available');
    }
}

function fetchDataAndViewCharts(projectId, selectedFile) {
    console.log("Fetching data for charts...");
    fetch(`/data/get_histogram_and_parallel_coordinates_data/${projectId}/${encodeURIComponent(selectedFile)}/`)
        .then(response => {
            console.log("Response fetchDataAndViewCharts received:", response);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Data from server fetchDataAndViewCharts:", data);
            timeData = data.histogramData.time || {};
            timeData2d = data.histogram2dData.time || {};
            timeData3d = data.histogram3dData.time || {};
            geoData = data.histogramData.geographical || {};
            geoData2d = data.histogram2dData.geographical || {};
            geoData3d = data.histogram3dData.geographical || {};
            renderHistograms();
            renderParallelCoordinates(data.parallelCoordinatesData);

            // 生成数据量
            const dataCount = data.parallelCoordinatesData.length;

            // 传递数据量到 updateParallelCoordinatesTitle
            updateParallelCoordinatesTitle(dataCount);
        })
        .catch(error => {
            console.error('Error fetching data for charts:', error);
            showErrorMessage('无法获取图表数据，请检查控制台以获取更多信息。');
        });
}

function updateParallelCoordinatesTitle(dataCount, previousRadarChartData) {
    const selectedTargets = [];
    const previousSelectedTargets = [];
    const formData = new FormData(document.getElementById('trainingForm'));
    
    formData.forEach((value, key) => {
        if (key.startsWith('features[') && key.endsWith(']') && value === 'target_variable') {
            const columnName = key.split('[')[1].split(']')[0];
            selectedTargets.push(columnName);
        }
    });

    if (previousRadarChartData && previousRadarChartData.length > 0) {
        previousRadarChartData[0].target.forEach(target => {
            previousSelectedTargets.push(`Previous ${target}`);
        });
    }

    const parallelCoordinatesTitle = document.querySelector('.parallel-coordinates h4');
    if (parallelCoordinatesTitle) {
        parallelCoordinatesTitle.innerHTML = `Parallel coordinate plot&nbsp;&nbsp;&nbsp;&nbsp;Sum：${dataCount}&nbsp;&nbsp;&nbsp;&nbsp;Target Variables:`; 

        // 添加新的目标变量
        selectedTargets.forEach((target, index) => {
            const span = createTargetSpan(target, defaultColors[index % defaultColors.length]);
            parallelCoordinatesTitle.appendChild(span);
        });

        // 添加 previous 目标变量
        previousSelectedTargets.forEach((target, index) => {
            const span = createTargetSpan(target, previousColors[index % previousColors.length]);
            parallelCoordinatesTitle.appendChild(span);
        });
    }
}

function createTargetSpan(target, color) {
    const span = document.createElement('span');
    span.textContent = ` ${target}`;
    span.style.marginLeft = '10px';
    span.style.cursor = 'pointer';
    span.style.color = color;
    span.setAttribute('data-target', target);
    span.onclick = () => handleTargetClick(target);
    return span;
}



function fetchDataAndRenderCharts(projectId, selectedFile) {
    console.log("Fetching data for charts...");
    fetch(`/data/get_data_for_charts/${projectId}/${encodeURIComponent(selectedFile)}/`)
        .then(response => {
            console.log("Response fetchDataAndRenderCharts received:", response);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Data from server fetchDataAndRenderCharts:", data);
            renderRadarChart(data.radarChartData, data.previousRadarChartData);
            renderLossChart(data.lossChartData, data.previousLossChartData);
            renderImportanceChart(data.importanceChartData);
            renderPredictionChart(data.predictionChartData, data.previousPredictionChartData);

            // 生成数据量
            let dataCount = 0;
            if (Array.isArray(data.predictionChartData) && data.predictionChartData.length > 0) {
                const firstItem = data.predictionChartData[0];
                if (Array.isArray(firstItem.y_true)) {
                    dataCount = firstItem.y_true.length;
                }
            }

            // 获取训练集比例
            const testSizeRatioInput = document.getElementById('testSizeRatio');
            const testSizeRatio = parseFloat(testSizeRatioInput.value);


            // 计算实际数据量
            const actualDataCount = dataCount / testSizeRatio;

            // 传递数据量到 updateParallelCoordinatesTitle
            updateParallelCoordinatesTitle(actualDataCount, data.previousRadarChartData);
        })
        .catch(error => {
            console.error('Error fetching data for charts:', error);
            showErrorMessage('无法获取图表数据，请检查控制台以获取更多信息。');
        });
}



function startBayesianOptimization() {
    console.log("Opening Bayesian Optimization modal...");
    $('#bayesianOptimizationModal').modal('show');
}

function startBayesianOptimizationWithModal() {
    console.log("Starting Bayesian Optimization with modal...");
    const n_calls = parseInt(document.getElementById('bayesianOptimizationCalls').value);
    const w1 = parseFloat(document.getElementById('w1').value);
    const w2 = parseFloat(document.getElementById('w2').value);
    const w3 = parseFloat(document.getElementById('w3').value);
    $('#bayesianOptimizationModal').modal('hide');

    console.log("Starting Bayesian Optimization...");
    const progressBar = document.getElementById('trainingProgress');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';

    const formData = new FormData(document.getElementById('trainingForm'));
    const projectId = getProjectId();

    formData.append('n_calls', n_calls);
    formData.append('w1', w1);
    formData.append('w2', w2);
    formData.append('w3', w3);

    fetch(`/data/bayesian_optimization/${projectId}/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
        },
        body: formData
    })
    .then(response => {
        console.log("Response received:", response);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Data from server:", data);
        if (data.status === 'success') {
            clearInterval(trainingInterval);
            document.getElementById('saveButton').disabled = false;
            console.log("Bayesian Optimization completed.");
            const bestParams = data.best_params;
            const score = data.score;  // 获取评分
            showParameterUpdateModal(bestParams, score);  // 传递评分
        } else {
            showErrorMessage(data.message);
        }
    })
    .catch(error => {
        console.error('Error starting Bayesian Optimization:', error);
        showErrorMessage('发生错误，请检查控制台以获取更多信息。');
    });
}

function showParameterUpdateModal(bestParams, score) {
    const modalBody = document.getElementById('parameterUpdateModalBody');
    modalBody.innerHTML = '';

    // 添加标题
    const title = document.createElement('h4');
    title.textContent = '最优参数组合';
    modalBody.appendChild(title);

    // 添加隐藏的输入字段来存储 bestParams
    const bestParamsInput = document.createElement('input');
    bestParamsInput.type = 'hidden';
    bestParamsInput.id = 'bestParamsInput';
    bestParamsInput.value = JSON.stringify(bestParams);
    modalBody.appendChild(bestParamsInput);

    // 添加隐藏的输入字段来存储 score
    const scoreInput = document.createElement('input');
    scoreInput.type = 'hidden';
    scoreInput.id = 'scoreInput';
    scoreInput.value = score;
    modalBody.appendChild(scoreInput);

    const paramNames = [
        'Temporal Kernel Size', 'LSTM Units', 'Spatial Kernel Size', 
        'Filters', 'Dynamic Attention Units', 'Dense Units', 'Epochs', 'Batch Size'
    ];

    // 定义 paramIds 数组
    const paramIds = [
        'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
        'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
    ];

    const orderedParams = [
        bestParams[5], bestParams[2], bestParams[6], bestParams[4], bestParams[7], bestParams[3], bestParams[0], bestParams[1]
    ];

    orderedParams.forEach((value, index) => {
        const input = document.getElementById(paramIds[index]);
        const valueDisplay = document.getElementById(paramIds[index] + '_value');
        if (valueDisplay) {
            valueDisplay.textContent = value;
        }

        const div = document.createElement('div');
        div.className = 'form-group d-flex align-items-center';
        div.innerHTML = `
            <label class="mr-2 small-label">${paramNames[index]}: ${value}</label>
        `;
        modalBody.appendChild(div);
    });

    // 添加评分展示
    const scoreDiv = document.createElement('div');
    scoreDiv.className = 'form-group d-flex align-items-center';
    scoreDiv.innerHTML = `
        <label class="mr-2 small-label">综合评分: ${score.toFixed(2)}(越小越好)</label>
    `;
    modalBody.appendChild(scoreDiv);

    $('#parameterUpdateModal').modal('show');
}

function updateParameters(bestParams) {
    const paramIds = [
        'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
        'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
    ];

    const orderedParams = [
        bestParams[5], bestParams[2], bestParams[6], bestParams[4], bestParams[7], bestParams[3], bestParams[0], bestParams[1]
    ];

    orderedParams.forEach((value, index) => {
        const input = document.getElementById(paramIds[index]);
        const min = Math.max(1, value - 20);
        const max = value + 20;

        input.min = min;
        input.max = max;
        input.value = value;

        const minSpan = document.querySelector(`label[for="${paramIds[index]}"] .range-min`);
        const maxSpan = document.querySelector(`label[for="${paramIds[index]}"] .range-max`);

        if (minSpan && maxSpan) {
            minSpan.textContent = min;
            maxSpan.textContent = max;
        }
    });
}

document.getElementById('updateParametersButton').addEventListener('click', function() {
    const bestParamsInput = document.getElementById('bestParamsInput');
    if (bestParamsInput) {
        const bestParams = JSON.parse(bestParamsInput.value);
        updateParameters(bestParams);
    }
    $('#parameterUpdateModal').modal('hide');
    // 将焦点设置到某个可聚焦的元素上，例如 bayesianOptimizationButton
    document.getElementById('bayesianOptimizationButton').focus();
});

document.getElementById('keepParametersButton').addEventListener('click', function() {
    const bestParamsInput = document.getElementById('bestParamsInput');
    if (bestParamsInput) {
        const bestParams = JSON.parse(bestParamsInput.value);
        const paramIds = [
            'temporal_kernel_size', 'lstm_units', 'spatial_kernel_sizes', 
            'filters', 'dynamic_attention_units', 'dense_units', 'epochs', 'batch_size'
        ];

        const orderedParams = [
            bestParams[5], bestParams[2], bestParams[6], bestParams[4], bestParams[7], bestParams[3], bestParams[0], bestParams[1]
        ];

        orderedParams.forEach((value, index) => {
            const input = document.getElementById(paramIds[index]);
            const valueDisplay = document.getElementById(paramIds[index] + '_value');
            if (valueDisplay) {
                valueDisplay.textContent = value;
            }
            input.value = value;
        });
    }
    $('#parameterUpdateModal').modal('hide');
    // 将焦点设置到某个可聚焦的元素上，例如 bayesianOptimizationButton
    document.getElementById('bayesianOptimizationButton').focus();
});

// 添加事件监听器来更新滑动条的数值显示
document.addEventListener('DOMContentLoaded', () => {
    const rangeInputs = document.querySelectorAll('.form-control-range');
    rangeInputs.forEach(input => {
        input.addEventListener('input', function() {
            const valueDisplay = document.getElementById(this.id + '_value');
            valueDisplay.textContent = this.value;
        });
    });

    const bayesianOptimizationCallsInput = document.getElementById('bayesianOptimizationCalls');
    bayesianOptimizationCallsInput.addEventListener('input', function() {
        const valueDisplay = document.getElementById(this.id + '_value');
        valueDisplay.textContent = this.value;
    });
    
});























