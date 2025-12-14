export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  FPR: number;  // False Positive Rate instead of f1_score
  auc: number;
}

export interface PredictionResult {
  predicted_label: string;
  confidence: number;
  threat_level: 'High' | 'Medium' | 'Low' | 'None';
  probabilities?: Record<string, number>;  // 各类别概率分布
  timestamp?: string;  // 检测时间戳
  encoded_value?: number;  // 编码值
}

export interface TrafficData {
  features: number[];
}

export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: 'success' | 'danger' | 'info';
  confidence?: string;
  label?: string;
  threat_level?: string;  // 威胁等级
  probabilities?: Record<string, number>;  // 概率分布
}

export interface ApiError {
  message: string;
}
