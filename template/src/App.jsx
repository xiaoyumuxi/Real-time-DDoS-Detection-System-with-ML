import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TrafficChart from './components/TrafficChart';
import AttackStats from './components/AttackStats';
import AlertList from './components/AlertList';
import './styles/main.css';

function App() {
  // 状态管理
  const [trafficData, setTrafficData] = useState([]);
  const [attackStats, setAttackStats] = useState({
    normal: 0,
    ddos: 0,
    total: 0
  });
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // 从后端API获取数据
  const fetchData = async () => {
    try {
      setIsLoading(true);
      const [trafficResponse, statsResponse, alertsResponse] = await Promise.all([
        axios.get('http://localhost:5000/api/traffic'),
        axios.get('http://localhost:5000/api/stats'),
        axios.get('http://localhost:5000/api/alerts')
      ]);

      setTrafficData(trafficResponse.data);
      setAttackStats(statsResponse.data);
      setAlerts(alertsResponse.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('无法连接到后端服务，请确保后端已启动。');
      // 使用模拟数据以便在后端未启动时仍能显示界面
      setTrafficData(getMockTrafficData());
      setAttackStats({ normal: 1200, ddos: 80, total: 1280 });
      setAlerts(getMockAlerts());
    } finally {
      setIsLoading(false);
    }
  };

  // 模拟流量数据
  const getMockTrafficData = () => {
    const data = [];
    const now = new Date();
    for (let i = 60; i >= 0; i--) {
      const time = new Date(now - i * 60000); // 过去60分钟，每分钟一个点
      data.push({
        timestamp: time.toISOString(),
        packets: Math.floor(Math.random() * 200) + 100,
        bytes: Math.floor(Math.random() * 1000000) + 500000,
        is_attack: Math.random() > 0.9 // 10%的概率是攻击
      });
    }
    return data;
  };

  // 模拟警报数据
  const getMockAlerts = () => {
    return [
      {
        id: 1,
        timestamp: new Date(Date.now() - 300000).toISOString(),
        src_ip: '192.168.1.100',
        dst_ip: '192.168.1.200',
        attack_type: 'DDoS',
        status: 'warning'
      },
      {
        id: 2,
        timestamp: new Date(Date.now() - 600000).toISOString(),
        src_ip: '10.0.0.50',
        dst_ip: '10.0.0.100',
        attack_type: 'DDoS',
        status: 'warning'
      },
      {
        id: 3,
        timestamp: new Date(Date.now() - 1200000).toISOString(),
        src_ip: '172.16.0.75',
        dst_ip: '172.16.0.25',
        attack_type: 'Normal',
        status: 'normal'
      }
    ];
  };

  // 初始加载数据
  useEffect(() => {
    fetchData();
    // 定期刷新数据（每30秒）
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  // 手动刷新数据
  const handleRefresh = () => {
    fetchData();
  };

  return (
    <div className="app">
      {/* 顶部导航栏 */}
      <header className="app-header">
        <h1>人工智能入侵检测系统</h1>
        <button className="refresh-btn" onClick={handleRefresh}>
          {isLoading ? '刷新中...' : '刷新数据'}
        </button>
      </header>

      {/* 主要内容区域 */}
      <main className="app-main">
        {/* 错误提示 */}
        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        {/* 攻击统计卡片 */}
        <section className="stats-section">
          <AttackStats stats={attackStats} />
        </section>

        {/* 流量图表 */}
        <section className="chart-section">
          <h2>网络流量实时监控</h2>
          <TrafficChart data={trafficData} />
        </section>

        {/* 警报列表 */}
        <section className="alerts-section">
          <h2>实时警报</h2>
          <AlertList alerts={alerts} />
        </section>
      </main>

      {/* 页脚 */}
      <footer className="app-footer">
        <p>© 2023 人工智能入侵检测系统 - 基于机器学习的DDoS攻击检测</p>
      </footer>
    </div>
  );
}

export default App;