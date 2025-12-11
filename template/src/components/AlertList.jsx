import React from 'react';
import '../styles/main.css';

const AlertList = ({ alerts }) => {
  // 格式化时间戳
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // 根据攻击类型获取状态样式
  const getStatusClass = (attackType) => {
    switch (attackType.toLowerCase()) {
      case 'ddos':
        return 'status-ddos';
      case 'normal':
        return 'status-normal';
      default:
        return 'status-warning';
    }
  };

  return (
    <div className="alerts-container">
      {alerts.length === 0 ? (
        <div className="no-alerts">
          <p>暂无警报信息</p>
        </div>
      ) : (
        <table className="alerts-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>源IP</th>
              <th>目标IP</th>
              <th>攻击类型</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((alert) => (
              <tr key={alert.id} className={getStatusClass(alert.attack_type)}>
                <td>{formatTime(alert.timestamp)}</td>
                <td>{alert.src_ip}</td>
                <td>{alert.dst_ip}</td>
                <td>{alert.attack_type}</td>
                <td>
                  <span className={`status-badge ${getStatusClass(alert.attack_type)}`}>
                    {alert.attack_type === 'Normal' ? '正常' : '警告'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default AlertList;