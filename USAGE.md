# 人工智能入侵检测系统使用说明

## 项目概述

本项目是一个基于机器学习的入侵检测系统，主要用于检测DDoS攻击。系统使用CICIDS2017数据集进行训练，并提供了模拟的API接口和前端页面来展示检测结果。

## 系统架构

1. **数据处理模块**：负责加载和清洗CICIDS2017数据集
2. **模型训练模块**：使用随机森林算法训练入侵检测模型
3. **后端API模块**：提供模拟的API接口用于预测和数据查询
4. **前端展示模块**：提供可视化界面展示检测结果和警报

## 快速开始

### 1. 运行后端API服务器

```bash
python simple_server.py
```

服务器将在端口5000上运行，提供以下API接口：
- `http://localhost:5000/api/health` - 健康检查
- `http://localhost:5000/api/sample` - 获取样本特征数据
- `http://localhost:5000/api/predict` - 进行攻击预测
- `http://localhost:5000/api/alerts` - 获取警报列表
- `http://localhost:5000/api/stats` - 获取统计信息
- `http://localhost:5000/api/traffic` - 获取流量数据

### 2. 运行前端服务器

```bash
python -m http.server 8000
```

前端页面将在端口8000上运行，可以通过以下地址访问：
- `http://localhost:8000/simple_frontend.html` - 简化版前端页面

## 核心功能

### 1. 数据处理

数据处理模块用于加载和清洗CICIDS2017数据集：

```bash
python clean.py
```

### 2. 模型训练

模型训练模块使用随机森林算法训练入侵检测模型：

```bash
python trainning.py
```

训练完成后，模型将保存为以下文件：
- `ddos_rf_model.joblib` - 随机森林模型
- `ddos_scaler.joblib` - 特征缩放器
- `ddos_label_encoder.joblib` - 标签编码器
- `ddos_feature_columns.joblib` - 特征列名称

### 3. 攻击预测

可以使用API接口进行攻击预测：

```bash
curl -X POST -H "Content-Type: application/json" -d '{"features": [54865, 3, 2, 0, 12, 0, 6, 6, 6.0, 0.0, 0, 0, 0.0, 0.0, 4000000.0, 666666.6667, 3.0, 0.0, 3, 3, 3, 3.0, 0.0, 3, 3, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 40, 0, 666666.6667, 0.0, 6, 6, 6.0, 0.0, 0.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 9.0, 6.0, 0.0, 40, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0, 33, -1, 1, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0]}' http://localhost:5000/api/predict
```

### 4. 实时监控

通过前端页面可以实时监控网络流量和攻击警报：
- 显示正常流量和DDoS攻击的统计信息
- 绘制流量变化图表
- 展示实时警报列表

## 文件说明

- `clean.py` - 数据清洗脚本
- `trainning.py` - 模型训练脚本
- `simple_server.py` - 模拟后端API服务器
- `simple_frontend.html` - 简化版前端页面
- `ddos_rf_model.joblib` - 训练好的随机森林模型
- `ddos_scaler.joblib` - 特征缩放器
- `ddos_label_encoder.joblib` - 标签编码器
- `ddos_feature_columns.joblib` - 特征列名称

## 技术栈

- **Python** - 后端开发语言
- **scikit-learn** - 机器学习库
- **Chart.js** - 数据可视化库
- **Bootstrap** - 前端UI框架

## 注意事项

1. 由于环境限制，本项目使用了模拟的API接口和简化的前端页面
2. 实际部署时，建议使用真实的Flask后端和React前端
3. 系统可以通过重新运行trainning.py脚本进行模型更新
4. 可以通过修改simple_server.py来调整API接口的行为

## 扩展建议

1. 添加更多类型的攻击检测算法
2. 实现真实的网络流量捕获和分析
3. 添加邮件或短信警报功能
4. 实现用户认证和权限管理
5. 优化模型性能和响应速度