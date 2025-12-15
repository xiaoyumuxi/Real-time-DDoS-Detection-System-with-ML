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
  probabilities?: Record<string, number>;  // Probability distributions of various categories
  timestamp?: string;  // Detection of timestamp
  encoded_value?: number;  // Code value
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
  threat_level?: string;  // Threat level
  probabilities?: Record<string, number>;  // Probability distribution
}

export interface ApiError {
  message: string;
}
