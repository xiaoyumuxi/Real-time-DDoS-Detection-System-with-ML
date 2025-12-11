# 人工智能入侵检测系统 - 前端应用

## 项目简介

这是一个基于React的前端应用，用于展示人工智能入侵检测系统的实时监控数据。该应用能够接收后端传来的网络流量数据、攻击统计信息和警报信息，并通过直观的图表和界面进行展示。

## 技术栈

- **前端框架**: React 18
- **构建工具**: Vite 5
- **HTTP客户端**: Axios
- **图表库**: Chart.js + react-chartjs-2
- **样式**: CSS3 (亮色主题)

## 功能特性

1. **实时流量监控**: 通过折线图展示网络流量的变化趋势，区分正常流量和攻击流量
2. **攻击统计分析**: 显示总流量、正常流量、DDoS攻击流量的统计信息和风险等级
3. **实时警报列表**: 展示最新的网络攻击警报信息，包括时间、源IP、目标IP、攻击类型等
4. **数据自动刷新**: 每30秒自动从后端获取最新数据，确保信息实时性
5. **响应式设计**: 适配不同屏幕尺寸的设备

## 安装和运行

### 前提条件

- Node.js (版本 14.x 或更高)
- npm (版本 6.x 或更高)

### 安装步骤

1. 进入前端项目目录

```bash
cd template
```

2. 安装依赖包

```bash
npm install
```

### 运行项目

#### 开发模式

```bash
npm run dev
```

应用将在 `http://localhost:5173` 启动

#### 生产构建

```bash
npm run build
```

构建后的文件将输出到 `dist` 目录

#### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
template/
├── public/              # 静态资源文件
├── src/
│   ├── components/      # React组件
│   │   ├── TrafficChart.jsx    # 流量图表组件
│   │   ├── AttackStats.jsx     # 攻击统计组件
│   │   └── AlertList.jsx       # 警报列表组件
│   ├── styles/          # 样式文件
│   │   └── main.css     # 主样式文件
│   ├── App.jsx          # 主应用组件
│   └── main.jsx         # 应用入口文件
├── index.html           # HTML模板
├── package.json         # 项目配置和依赖
└── README.md            # 项目说明文档
```

## API接口说明

前端应用需要与后端API进行交互，以下是所需的API接口：

### 1. 获取流量数据

```
GET http://localhost:5000/api/traffic
```

响应示例：
```json
[
  {
    "timestamp": "2023-10-15T10:00:00Z",
    "packets": 150,
    "bytes": 750000,
    "is_attack": false
  },
  {
    "timestamp": "2023-10-15T10:01:00Z",
    "packets": 250,
    "bytes": 1250000,
    "is_attack": true
  }
]
```

### 2. 获取攻击统计

```
GET http://localhost:5000/api/stats
```

响应示例：
```json
{
  "normal": 1200,
  "ddos": 80,
  "total": 1280
}
```

### 3. 获取警报列表

```
GET http://localhost:5000/api/alerts
```

响应示例：
```json
[
  {
    "id": 1,
    "timestamp": "2023-10-15T10:05:00Z",
    "src_ip": "192.168.1.100",
    "dst_ip": "192.168.1.200",
    "attack_type": "DDoS",
    "status": "warning"
  }
]
```

## 模拟数据

如果后端服务未启动，应用将使用模拟数据进行展示，确保界面能够正常运行。

## 样式主题

本应用采用亮色主题设计，主要特点：

- 背景色：浅灰色 (#f5f7fa)
- 卡片背景：白色 (#ffffff)
- 主色调：蓝色 (#3498db)
- 成功色：绿色 (#27ae60)
- 警告色：红色 (#e74c3c)
- 风险色：橙色 (#f39c12)

## 浏览器支持

- Chrome (最新版本)
- Firefox (最新版本)
- Safari (最新版本)
- Edge (最新版本)

## 开发说明

### 添加新组件

1. 在 `src/components/` 目录下创建新的组件文件
2. 在需要使用的组件中导入并使用

### 修改样式

- 全局样式：修改 `src/styles/main.css`
- 组件样式：可以在组件文件中使用内联样式或创建单独的CSS文件

### 调试技巧

1. 使用浏览器的开发者工具查看组件状态和网络请求
2. 在 `App.jsx` 中设置 `console.log` 查看数据流转
3. 使用React DevTools扩展进行组件调试

## 注意事项

1. 确保后端服务运行在 `http://localhost:5000`
2. 如果后端API地址或端口发生变化，请修改 `App.jsx` 中的API请求地址
3. 首次运行前请确保已安装所有依赖包
4. 生产环境部署前请执行 `npm run build` 进行构建

## License

MIT