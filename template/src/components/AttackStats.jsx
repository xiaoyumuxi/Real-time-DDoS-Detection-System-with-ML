import React from 'react';
import '../styles/main.css';

const AttackStats = ({ stats }) => {
  // 计算百分比
  const normalPercentage = stats.total > 0 ? Math.round((stats.normal / stats.total) * 100) : 0;
  const ddosPercentage = stats.total > 0 ? Math.round((stats.ddos / stats.total) * 100) : 0;

  return (
    <div className="stats-container">
      <div className="stat-card total">
        <h3>总流量</h3>
        <div className="stat-value">{stats.total}</div>
        <div className="stat-label">数据包</div>
      </div>
      
      <div className="stat-card normal">
        <h3>正常流量</h3>
        <div className="stat-value">{stats.normal}</div>
        <div className="stat-label">{normalPercentage}% of total</div>
      </div>
      
      <div className="stat-card attack">
        <h3>DDoS攻击</h3>
        <div className="stat-value">{stats.ddos}</div>
        <div className="stat-label">{ddosPercentage}% of total</div>
      </div>
      
      <div className="stat-card risk">
        <h3>风险等级</h3>
        <div className="risk-indicator">
          {ddosPercentage < 5 ? (
            <div className="risk-level low">低</div>
          ) : ddosPercentage < 20 ? (
            <div className="risk-level medium">中</div>
          ) : (
            <div className="risk-level high">高</div>
          )}
        </div>
        <div className="stat-label">基于当前攻击比例</div>
      </div>
    </div>
  );
};

export default AttackStats;