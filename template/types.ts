export interface PredictionResult {
    status: string;
    predicted_label: string;
    confidence: number;
    encoded_value: number;
    threat_level: 'High' | 'Medium' | 'Low' | 'None';
    message?: string;
}

export interface AlertLog {
    timestamp: string;
    type: string;
    confidence: number;
    level: 'High' | 'Medium' | 'Low' | 'None';
}

export interface PerformanceMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc: number;
}