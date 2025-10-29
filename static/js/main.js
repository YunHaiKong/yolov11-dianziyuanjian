// 主要的JavaScript功能

// 全局变量
let currentPage = 'index';
let charts = {};
let websocket = null;

// 页面加载完成后初始化
$(document).ready(function() {
    initializeApp();
    setupEventListeners();
    startWebSocket();
});

// 初始化应用
function initializeApp() {
    // 设置当前页面
    const path = window.location.pathname;
    if (path.includes('training')) {
        currentPage = 'training';
    } else if (path.includes('prediction')) {
        currentPage = 'prediction';
    } else if (path.includes('analysis')) {
        currentPage = 'analysis';
    } else if (path.includes('comparison')) {
        currentPage = 'comparison';
    }
    
    // 设置活动导航项
    $('.nav-link').removeClass('active');
    $(`.nav-link[href*="${currentPage}"]`).addClass('active');
    
    // 初始化工具提示
    $('[data-bs-toggle="tooltip"]').tooltip();
    
    // 初始化页面特定功能
    switch(currentPage) {
        case 'index':
            initializeIndex();
            break;
        case 'training':
            initializeTraining();
            break;
        case 'prediction':
            initializePrediction();
            break;
        case 'analysis':
            initializeAnalysis();
            break;
        case 'comparison':
            initializeComparison();
            break;
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 侧边栏切换（移动端）
    $('#sidebar-toggle').on('click', function() {
        $('.sidebar').toggleClass('show');
    });
    
    // 点击外部关闭侧边栏（移动端）
    $(document).on('click', function(e) {
        if ($(window).width() <= 768) {
            if (!$(e.target).closest('.sidebar, #sidebar-toggle').length) {
                $('.sidebar').removeClass('show');
            }
        }
    });
    
    // 文件上传拖拽
    setupDragAndDrop();
    
    // 全局错误处理
    $(document).ajaxError(function(event, xhr, settings, error) {
        console.error('AJAX Error:', error);
        showNotification('网络请求失败，请检查连接', 'error');
    });
}

// 初始化首页
function initializeIndex() {
    loadSystemStatus();
    loadRecentTraining();
    
    // 定期更新状态
    setInterval(loadSystemStatus, 30000); // 30秒更新一次
}

// 初始化训练页面
function initializeTraining() {
    loadTrainingRecords();
}

// 初始化预测页面
function initializePrediction() {
    loadAvailableModels();
    setupImageUpload();
    loadPredictionHistory();
}

// 初始化分析页面
function initializeAnalysis() {
    // 分析页面的初始化在analysis.html中处理
}

// 初始化比较页面
function initializeComparison() {
    // 比较页面的初始化在comparison.html中处理
}

// 加载系统状态
function loadSystemStatus() {
    $.get('/api/system_status', function(data) {
        updateSystemStatus(data);
    }).fail(function() {
        console.error('Failed to load system status');
    });
}

// 更新系统状态显示
function updateSystemStatus(data) {
    $('#training-projects').text(data.training_projects || 0);
    $('#available-models').text(data.available_models || 0);
    $('#datasets').text(data.datasets || 0);
    
    // 更新系统状态指示器
    const statusIndicator = $('.status-indicator');
    if (data.system_status === 'online') {
        statusIndicator.removeClass('offline warning').addClass('online');
        $('#system-status').text('正常运行');
    } else if (data.system_status === 'warning') {
        statusIndicator.removeClass('online offline').addClass('warning');
        $('#system-status').text('警告状态');
    } else {
        statusIndicator.removeClass('online warning').addClass('offline');
        $('#system-status').text('离线状态');
    }
}

// 加载最近训练记录
function loadRecentTraining() {
    $.get('/api/recent_training', function(data) {
        updateRecentTraining(data);
    }).fail(function() {
        console.error('Failed to load recent training');
    });
}

// 更新最近训练记录显示
function updateRecentTraining(data) {
    const container = $('#recent-training');
    if (!data.records || data.records.length === 0) {
        container.html('<p class="text-muted">暂无训练记录</p>');
        return;
    }
    
    let html = '';
    data.records.slice(0, 5).forEach(record => {
        const statusClass = record.status === 'completed' ? 'success' : 
                           record.status === 'running' ? 'primary' : 'warning';
        const statusText = record.status === 'completed' ? '已完成' : 
                          record.status === 'running' ? '运行中' : '已停止';
        
        html += `
            <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                <div>
                    <strong>${record.name}</strong><br>
                    <small class="text-muted">${formatDate(record.created_at)}</small>
                </div>
                <div class="text-end">
                    <span class="badge bg-${statusClass}">${statusText}</span><br>
                    <small class="text-muted">mAP: ${(record.map || 0).toFixed(3)}</small>
                </div>
            </div>
        `;
    });
    
    container.html(html);
}

// 加载训练记录
function loadTrainingRecords() {
    $.get('/api/training_records', function(data) {
        updateTrainingRecords(data);
    }).fail(function() {
        console.error('Failed to load training records');
    });
}

// 更新训练记录显示
function updateTrainingRecords(data) {
    const container = $('#training-records');
    if (!data.records || data.records.length === 0) {
        container.html('<p class="text-muted text-center">暂无训练记录</p>');
        return;
    }
    
    let html = '';
    data.records.forEach(record => {
        const statusClass = record.status === 'completed' ? 'success' : 
                           record.status === 'running' ? 'primary' : 'warning';
        const statusText = record.status === 'completed' ? '已完成' : 
                          record.status === 'running' ? '运行中' : '已停止';
        
        html += `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h6 class="card-title">${record.name}</h6>
                            <span class="badge bg-${statusClass}">${statusText}</span>
                        </div>
                        <p class="card-text">
                            <small class="text-muted">创建时间: ${formatDate(record.created_at)}</small><br>
                            <small class="text-muted">数据集: ${record.dataset || 'N/A'}</small><br>
                            <small class="text-muted">轮数: ${record.epochs || 'N/A'}</small>
                        </p>
                        <div class="row text-center">
                            <div class="col-4">
                                <small class="text-muted">mAP50</small><br>
                                <strong>${(record.map50 || 0).toFixed(3)}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted">mAP95</small><br>
                                <strong>${(record.map95 || 0).toFixed(3)}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted">精确率</small><br>
                                <strong>${(record.precision || 0).toFixed(3)}</strong>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-primary" onclick="viewTrainingDetails('${record.id}')">
                                <i class="fas fa-eye"></i> 查看详情
                            </button>
                            <button class="btn btn-sm btn-success" onclick="downloadModel('${record.id}')">
                                <i class="fas fa-download"></i> 下载模型
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    container.html(html);
}

// 查看训练详情
function viewTrainingDetails(recordId) {
    $.get(`/api/training_details/${recordId}`, function(data) {
        showTrainingDetailsModal(data);
    }).fail(function() {
        showNotification('无法加载训练详情', 'error');
    });
}

// 显示训练详情模态框
function showTrainingDetailsModal(data) {
    const modalHtml = `
        <div class="modal fade" id="trainingDetailsModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">训练详情 - ${data.name}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>基本信息</h6>
                                <table class="table table-sm">
                                    <tr><td>状态</td><td><span class="badge bg-success">${data.status}</span></td></tr>
                                    <tr><td>数据集</td><td>${data.dataset}</td></tr>
                                    <tr><td>模型</td><td>${data.model}</td></tr>
                                    <tr><td>训练轮数</td><td>${data.epochs}</td></tr>
                                    <tr><td>批次大小</td><td>${data.batch_size}</td></tr>
                                    <tr><td>学习率</td><td>${data.learning_rate}</td></tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h6>性能指标</h6>
                                <table class="table table-sm">
                                    <tr><td>mAP@0.5</td><td>${(data.metrics.map50 || 0).toFixed(3)}</td></tr>
                                    <tr><td>mAP@0.5:0.95</td><td>${(data.metrics.map95 || 0).toFixed(3)}</td></tr>
                                    <tr><td>精确率</td><td>${(data.metrics.precision || 0).toFixed(3)}</td></tr>
                                    <tr><td>召回率</td><td>${(data.metrics.recall || 0).toFixed(3)}</td></tr>
                                    <tr><td>F1分数</td><td>${(data.metrics.f1 || 0).toFixed(3)}</td></tr>
                                </table>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-12">
                                <h6>训练曲线</h6>
                                <canvas id="detailsChart" width="400" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                        <button type="button" class="btn btn-primary" onclick="downloadModel('${data.id}')">下载模型</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 移除现有模态框
    $('#trainingDetailsModal').remove();
    
    // 添加新模态框
    $('body').append(modalHtml);
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('trainingDetailsModal'));
    modal.show();
    
    // 绘制训练曲线
    setTimeout(() => {
        drawDetailsChart(data.training_curves);
    }, 500);
}

// 绘制详情图表
function drawDetailsChart(curves) {
    const ctx = document.getElementById('detailsChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: curves.epochs || [],
            datasets: [
                {
                    label: '训练损失',
                    data: curves.train_loss || [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: false
                },
                {
                    label: '验证损失',
                    data: curves.val_loss || [],
                    borderColor: '#fd7e14',
                    backgroundColor: 'rgba(253, 126, 20, 0.1)',
                    fill: false
                },
                {
                    label: 'mAP',
                    data: curves.map || [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '损失值'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'mAP值'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// 下载模型
function downloadModel(recordId) {
    window.open(`/api/download_model/${recordId}`, '_blank');
}

// 加载可用模型
function loadAvailableModels() {
    $.get('/api/models', function(data) {
        updateModelSelect(data.models);
    }).fail(function() {
        console.error('Failed to load models');
    });
}

// 更新模型选择器
function updateModelSelect(models) {
    const select = $('#model-select');
    select.empty().append('<option value="">选择模型...</option>');
    
    models.forEach(model => {
        select.append(`<option value="${model.path}">${model.name} (${model.type})</option>`);
    });
}

// 设置图片上传
function setupImageUpload() {
    const uploadArea = $('#upload-area');
    const fileInput = $('#image-file');
    const preview = $('#image-preview');
    
    // 点击上传区域
    uploadArea.on('click', function() {
        fileInput.click();
    });
    
    // 文件选择
    fileInput.on('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });
}

// 设置拖拽上传
function setupDragAndDrop() {
    const uploadArea = $('#upload-area');
    
    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('dragover');
    });
    
    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
    });
    
    uploadArea.on('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
        
        const files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });
}

// 处理图片上传
function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('请选择图片文件', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        $('#image-preview').attr('src', e.target.result).show();
        $('#upload-text').hide();
    };
    reader.readAsDataURL(file);
}

// 执行预测
function performPrediction() {
    const modelPath = $('#model-select').val();
    const confidence = $('#confidence-threshold').val();
    const iou = $('#iou-threshold').val();
    const imageFile = $('#image-file')[0].files[0];
    
    if (!modelPath) {
        showNotification('请选择模型', 'error');
        return;
    }
    
    if (!imageFile) {
        showNotification('请选择图片', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('model', modelPath);
    formData.append('image', imageFile);
    formData.append('confidence', confidence);
    formData.append('iou', iou);
    
    // 显示加载状态
    $('#prediction-results').html(`
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">预测中...</span>
            </div>
            <p class="mt-2">正在进行预测...</p>
        </div>
    `);
    
    $.ajax({
        url: '/api/predict',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data) {
            displayPredictionResults(data);
            savePredictionHistory(data);
        },
        error: function() {
            $('#prediction-results').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i> 预测失败，请重试
                </div>
            `);
        }
    });
}

// 显示预测结果
function displayPredictionResults(data) {
    const resultsHtml = `
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-image"></i> 检测结果</h6>
                    </div>
                    <div class="card-body text-center">
                        <img src="data:image/jpeg;base64,${data.result_image}" 
                             class="img-fluid image-preview" alt="检测结果">
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-list"></i> 检测统计</h6>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <small class="text-muted">检测到的对象</small><br>
                            <strong class="h4">${data.detections.length}</strong>
                        </div>
                        <div class="mb-3">
                            <small class="text-muted">推理时间</small><br>
                            <strong>${data.inference_time.toFixed(2)}ms</strong>
                        </div>
                        <div class="mb-3">
                            <small class="text-muted">平均置信度</small><br>
                            <strong>${(data.avg_confidence * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-tags"></i> 检测详情</h6>
                    </div>
                    <div class="card-body">
                        ${data.detections.map(det => `
                            <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                                <div>
                                    <strong>${det.class}</strong><br>
                                    <small class="text-muted">置信度: ${(det.confidence * 100).toFixed(1)}%</small>
                                </div>
                                <div class="text-end">
                                    <small class="text-muted">
                                        ${det.bbox[0]}, ${det.bbox[1]}<br>
                                        ${det.bbox[2]}×${det.bbox[3]}
                                    </small>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    $('#prediction-results').html(resultsHtml);
}

// 保存预测历史
function savePredictionHistory(data) {
    let history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    
    const record = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        model: $('#model-select option:selected').text(),
        detections: data.detections.length,
        inference_time: data.inference_time,
        avg_confidence: data.avg_confidence
    };
    
    history.unshift(record);
    history = history.slice(0, 10); // 只保留最近10条
    
    localStorage.setItem('predictionHistory', JSON.stringify(history));
    loadPredictionHistory();
}

// 加载预测历史
function loadPredictionHistory() {
    const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    const container = $('#prediction-history');
    
    if (history.length === 0) {
        container.html('<p class="text-muted">暂无预测历史</p>');
        return;
    }
    
    let html = '';
    history.forEach(record => {
        html += `
            <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                <div>
                    <strong>${record.model}</strong><br>
                    <small class="text-muted">${formatDate(record.timestamp)}</small>
                </div>
                <div class="text-end">
                    <span class="badge bg-primary">${record.detections} 个对象</span><br>
                    <small class="text-muted">${record.inference_time.toFixed(2)}ms</small>
                </div>
            </div>
        `;
    });
    
    container.html(html);
}

// 启动WebSocket连接
function startWebSocket() {
    if (websocket) {
        websocket.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function() {
        console.log('WebSocket connected');
    };
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onclose = function() {
        console.log('WebSocket disconnected');
        // 5秒后重连
        setTimeout(startWebSocket, 5000);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// 处理WebSocket消息
function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'training_update':
            updateTrainingProgress(data.data);
            break;
        case 'system_status':
            updateSystemStatus(data.data);
            break;
        case 'notification':
            showNotification(data.message, data.level);
            break;
    }
}

// 更新训练进度
function updateTrainingProgress(data) {
    if (currentPage === 'training') {
        // 更新训练页面的进度显示
        updateTrainingDisplay(data);
    }
}

// 显示通知
function showNotification(message, type = 'info', duration = 5000) {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const icon = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-triangle',
        'warning': 'fas fa-exclamation-circle',
        'info': 'fas fa-info-circle'
    }[type] || 'fas fa-info-circle';
    
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // 添加到通知容器
    let container = $('#notification-container');
    if (container.length === 0) {
        container = $('<div id="notification-container" style="position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;"></div>');
        $('body').append(container);
    }
    
    const alert = $(alertHtml);
    container.append(alert);
    
    // 自动关闭
    if (duration > 0) {
        setTimeout(() => {
            alert.alert('close');
        }, duration);
    }
}

// 格式化日期
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 导出函数供全局使用
window.showNotification = showNotification;
window.formatDate = formatDate;
window.formatFileSize = formatFileSize;
window.performPrediction = performPrediction;
window.viewTrainingDetails = viewTrainingDetails;
window.downloadModel = downloadModel;