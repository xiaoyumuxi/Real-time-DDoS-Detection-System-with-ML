export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc: number;
}

export interface PredictionResult {
  predicted_label: string;
  confidence: number;
  threat_level: 'High' | 'Medium' | 'Low' | 'None';
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
}

export interface ApiError {
  message: string;
}
