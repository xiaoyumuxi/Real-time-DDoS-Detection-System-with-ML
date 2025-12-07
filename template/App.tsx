import React, { useState, useEffect } from 'react';
import { PredictionResult, AlertLog, PerformanceMetrics } from './types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// API Base URL
const API_URL = 'http://127.0.0.1:5000/api';

const App: React.FC = () => {
    // State definition
    const [features, setFeatures] = useState<number[]>([]);
    const [featureNames, setFeatureNames] = useState<string[]>([]);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [alerts, setAlerts] = useState<AlertLog[]>([]);
    const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
    const [retraining, setRetraining] = useState<boolean>(false);

    // Initial Data Load
    useEffect(() => {
        fetchSampleData();
        fetchAlerts();
        fetchMetrics();
        // Set up polling for alerts every 5 seconds
        const interval = setInterval(fetchAlerts, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchSampleData = async () => {
        try {
            const res = await fetch(`${API_URL}/sample`);
            const data = await res.json();
            setFeatures(data.features);
            setFeatureNames(data.feature_names);
        } catch (error) {
            console.error("Error fetching sample:", error);
        }
    };

    const fetchAlerts = async () => {
        try {
            const res = await fetch(`${API_URL}/alerts`);
            const data = await res.json();
            setAlerts(data);
        } catch (error) {
            console.error("Error fetching alerts:", error);
        }
    };

    const fetchMetrics = async () => {
        try {
            const res = await fetch(`${API_URL}/performance`);
            const data = await res.json();
            setMetrics(data);
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });
            const data = await res.json();
            setResult(data);
            // Refresh alerts immediately if a threat was found
            if (data.threat_level !== 'None') {
                fetchAlerts();
            }
        } catch (error) {
            console.error("Prediction error:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleRetrain = async () => {
        if (!confirm("This will trigger the training script on the server. Continue?")) return;
        setRetraining(true);
        try {
            const res = await fetch(`${API_URL}/retrain`, { method: 'POST' });
            const data = await res.json();
            alert(data.message + (data.status === 'success' ? "\nPlease restart backend to load new model." : ""));
        } catch (error) {
            alert("Retraining failed check console.");
            console.error(error);
        } finally {
            setRetraining(false);
        }
    };

    // Helper for color coding alerts
    const getLevelColor = (level: string) => {
        switch (level) {
            case 'High': return 'bg-red-100 text-red-800 border-red-200';
            case 'Medium': return 'bg-orange-100 text-orange-800 border-orange-200';
            case 'Low': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            default: return 'bg-green-100 text-green-800 border-green-200';
        }
    };

    // Prepare chart data
    const chartData = metrics ? [
        { name: 'Accuracy', value: metrics.accuracy * 100 },
        { name: 'Precision', value: metrics.precision * 100 },
        { name: 'Recall', value: metrics.recall * 100 },
        { name: 'F1 Score', value: metrics.f1_score * 100 },
    ] : [];

    return (
        <div className="min-h-screen p-6 bg-slate-50 font-sans">
            <header className="mb-8 flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                <div>
                    <h1 className="text-3xl font-bold text-slate-800 tracking-tight">DDoS Defense Shield</h1>
                    <p className="text-slate-500 mt-1">Real-time Traffic Analysis & Threat Intelligence</p>
                </div>
                <div className="flex gap-4">
                     <button 
                        onClick={handleRetrain}
                        disabled={retraining}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${retraining ? 'bg-gray-300' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
                    >
                        {retraining ? 'Training...' : 'Retrain Model'}
                    </button>
                    <div className="px-4 py-2 bg-green-50 text-green-700 rounded-lg border border-green-100 font-medium">
                        System Active
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                {/* Left Column: Input & Prediction */}
                <div className="lg:col-span-8 space-y-8">
                    
                    {/* Prediction Panel */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold text-slate-800">Traffic Analyzer</h2>
                            <button 
                                onClick={fetchSampleData}
                                className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                            >
                                ↺ Reset to Sample Data
                            </button>
                        </div>
                        
                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 mb-6 max-h-48 overflow-y-auto">
                            <p className="text-xs text-slate-500 mb-2 font-mono">Raw Feature Vector ({features.length} features)</p>
                            <div className="flex flex-wrap gap-1 text-xs font-mono text-slate-600 break-all">
                                {features.join(', ')}
                            </div>
                        </div>

                        <button
                            onClick={handlePredict}
                            disabled={loading}
                            className={`w-full py-4 rounded-lg text-white font-bold text-lg shadow-lg transition-all transform active:scale-95 ${loading ? 'bg-slate-400' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl'}`}
                        >
                            {loading ? 'Analyzing Traffic...' : 'Analyze Traffic Pattern'}
                        </button>
                    </div>

                    {/* Result Display */}
                    {result && (
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 animate-fade-in">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Analysis Result</h2>
                            
                            <div className={`p-6 rounded-xl border-l-8 flex items-center justify-between ${result.predicted_label === 'BENIGN' ? 'bg-green-50 border-green-500' : 'bg-red-50 border-red-500'}`}>
                                <div>
                                    <h3 className="text-2xl font-extrabold uppercase tracking-wider mb-1">
                                        {result.predicted_label}
                                    </h3>
                                    <p className="text-slate-600">
                                        Confidence: <span className="font-mono font-bold">{(result.confidence * 100).toFixed(2)}%</span>
                                    </p>
                                </div>
                                <div className="text-right">
                                    <span className="block text-xs uppercase text-slate-500 font-bold mb-1">Threat Level</span>
                                    <span className={`inline-block px-4 py-1 rounded-full text-sm font-bold border ${getLevelColor(result.threat_level)}`}>
                                        {result.threat_level}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Performance Metrics Chart */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                         <h2 className="text-xl font-bold text-slate-800 mb-4">Model Performance (RF)</h2>
                         <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b'}} />
                                    <YAxis hide />
                                    <Tooltip 
                                        cursor={{fill: '#f1f5f9'}}
                                        contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} 
                                    />
                                    <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={50} label={{ position: 'top', fill: '#64748b', fontSize: 12 }} />
                                </BarChart>
                            </ResponsiveContainer>
                         </div>
                    </div>
                </div>

                {/* Right Column: Alert Logs */}
                <div className="lg:col-span-4 space-y-8">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 h-full max-h-[800px] flex flex-col">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-slate-800">Threat Intelligence Log</h2>
                            <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full font-bold animate-pulse">LIVE</span>
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                            {alerts.length === 0 ? (
                                <div className="text-center text-slate-400 py-10">
                                    <p>No threats detected yet.</p>
                                </div>
                            ) : (
                                alerts.map((alert, idx) => (
                                    <div key={idx} className="p-4 rounded-lg bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors">
                                        <div className="flex justify-between items-start mb-2">
                                            <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getLevelColor(alert.level)}`}>
                                                {alert.level} PRIORITY
                                            </span>
                                            <span className="text-xs text-slate-400">{alert.timestamp.split(' ')[1]}</span>
                                        </div>
                                        <p className="font-bold text-slate-800">{alert.type}</p>
                                        <p className="text-xs text-slate-500 mt-1">Confidence Score: {(alert.confidence * 100).toFixed(1)}%</p>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;