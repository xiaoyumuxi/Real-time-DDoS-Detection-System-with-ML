import { PerformanceMetrics, PredictionResult, TrafficData } from '../types';

const TIMEOUT_MS = 8000;

async function fetchWithTimeout(resource: string, options: RequestInit = {}): Promise<Response> {
  const { signal, ...rest } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS);

  // If a signal was passed, we need to respect it too, but for simplicity here
  // we primarily rely on our timeout controller.
  
  try {
    const response = await fetch(resource, {
      ...rest,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

export const api = {
  async getPerformance(): Promise<PerformanceMetrics> {
    const res = await fetchWithTimeout('/api/performance', { method: 'GET' });
    if (!res.ok) throw new Error("Server offline");
    return res.json();
  },

  async getTrafficData(type: 'normal' | 'attack' | 'random'): Promise<TrafficData> {
    let url = '/api/random';
    if (type === 'normal') url = '/api/sample';
    if (type === 'attack') url = '/api/simulate-attack';
    
    const res = await fetchWithTimeout(url);
    if (!res.ok) throw new Error("Failed to generate traffic data");
    return res.json();
  },

  async predict(features: number[]): Promise<PredictionResult> {
    const res = await fetchWithTimeout('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error("Prediction API failed");
    return res.json();
  },

  async retrain(file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('files', file);

    // Retraining needs a longer timeout
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch('/api/upload-and-retrain', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      clearTimeout(id);
      if (!res.ok) throw new Error("Retraining failed");
      return res.json();
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  }
};
