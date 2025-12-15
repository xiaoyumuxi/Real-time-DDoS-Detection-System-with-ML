import os
import logging
from datetime import datetime
from threading import Lock
import json
import pandas as pd
import numpy as np
import joblib
import sqlite3
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

# --- 配置部分 ---
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 上传最大 500MB
MAX_ALERTS = 50  # 内存中保存的最大警报数量
DB_FILE = 'ddos_detection.db'

# 全局变量 (模型组件)
MODEL = None
SCALER = None
LE = None
FEATURE_COLUMNS = None

# 模型文件路径 (与 trainning.py 保持一致)
MODEL_PATH = './models/ddos_rf_model.joblib'
SCALER_PATH = './models/ddos_scaler.joblib'
ENCODER_PATH = './models/ddos_label_encoder.joblib'
FEATURE_COLS_PATH = './models/ddos_feature_columns.joblib'
PERFORMANCE_PATH = './models/ddos_performance.json'

# 内存存储 (警报)
alerts = []
alerts_lock = Lock()

# 攻击样本库（用于 /api/stream 模拟攻击）
ATTACK_SAMPLE_LIBRARY = []          # [{'label': 'DoS Hulk', 'features': [...]}, ...]
attack_samples_lock = Lock()

# 从原始数据集中抽样构建攻击样本库的路径
ATTACK_DATASET_PATH = os.getenv(
    'ATTACK_DATASET_PATH',
    './data/Wednesday-workingHours.pcap_ISCX.csv'
)

# 频率统计时间窗（秒）
TIME_WINDOW_SECONDS = 10

# 性能指标 (示例初始值，训练后会更新)
PERFORMANCE_METRICS = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "auc": 0.0
}


# ----------------------------------------------------------------------
# 1. 数据库初始化
# ----------------------------------------------------------------------
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # 修复：添加 features_count 列，防止 INSERT 时报错
        c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                     (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,predicted_label TEXT NOT NULL,confidence REAL NOT NULL,threat_level TEXT NOT NULL,features_count INTEGER
                     )''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


init_db()


# ----------------------------------------------------------------------
# 2. 模型加载逻辑
# ----------------------------------------------------------------------
def load_model_components():
    """
    加载所有保存的模型组件
    """
    global MODEL, SCALER, LE, FEATURE_COLUMNS, PERFORMANCE_METRICS

    try:
        # 1. 加载模型二进制文件
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLS_PATH]):
            logger.warning("One or more model binary files not found.")
            return False

        MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        LE = joblib.load(ENCODER_PATH)
        FEATURE_COLUMNS = joblib.load(FEATURE_COLS_PATH)

        # 2. 加载性能指标 (新增逻辑)
        if os.path.exists(PERFORMANCE_PATH):
            try:
                with open(PERFORMANCE_PATH, 'r') as f:
                    PERFORMANCE_METRICS = json.load(f)
                logger.info("✅ Performance metrics loaded.")
            except Exception as e:
                logger.warning(f"⚠️ Found metrics file but failed to load: {e}")
        else:
            logger.warning("⚠️ No performance metrics file found (ddos_performance.json). Metrics will be 0.")

        logger.info("✅ Model components loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model files: {e}")
        return False


# ----------------------------------------------------------------------
# 3. 核心预测与辅助函数
# ----------------------------------------------------------------------
def get_threat_level(label, confidence):
    # todo: 这一部分需要重写逻辑
    """根据标签和置信度确定威胁等级"""
    if label.upper() == 'BENIGN':
        return 'None'
    elif confidence > 0.9:
        return 'High'
    elif confidence > 0.7:
        return 'Medium'
    else:
        return 'Low'


def predict(raw_input_data):
    """
    核心预测逻辑，与 api.py 保持一致
    """
    if not FEATURE_COLUMNS:
        return {"status": "error", "message": "Model not loaded."}

    # 1. 检查特征数量
    if len(raw_input_data) != len(FEATURE_COLUMNS):
        return {
            "status": "error",
            "message": f"Feature mismatch. Expected {len(FEATURE_COLUMNS)}, got {len(raw_input_data)}."
        }

    try:
        # 2. 转换为 DataFrame (使用保存的列名)
        new_df = pd.DataFrame([raw_input_data], columns=FEATURE_COLUMNS)

        # 3. 数据清理 (替换 Inf/NaN 为 0，确保健壮性)
        new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_df.fillna(0, inplace=True)

        # 4. 特征缩放
        data_scaled = SCALER.transform(new_df)

        # 5. 预测
        # 使用 NumPy 数组（而非带列名的 DataFrame）传入模型，避免 scikit-learn 关于
        # "X has feature names, but RandomForestClassifier was fitted without feature names" 的警告。
        prediction_encoded = MODEL.predict(data_scaled)[0]
        prediction_proba = MODEL.predict_proba(data_scaled)[0]

        # 6. 解析结果
        prediction_label = LE.inverse_transform([prediction_encoded])[0]
        max_proba = np.max(prediction_proba)
        threat_level = get_threat_level(prediction_label, max_proba)

        # 获取所有类别的概率分布（前5个最高概率）
        proba_dict = {}
        classes = LE.classes_
        for i, prob in enumerate(prediction_proba):
            if prob > 0.01:  # 只显示概率大于1%的
                proba_dict[classes[i]] = float(prob)
        
        # 按概率降序排序，取前5个
        top_probabilities = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        return {
            "status": "success",
            "predicted_label": prediction_label,
            "confidence": float(max_proba),
            "encoded_value": int(prediction_encoded),
            "threat_level": threat_level,
            "probabilities": top_probabilities,  # 所有类别的概率分布
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 添加时间戳
        }
    except Exception as e:
        logger.error(f"Prediction logic error: {e}")
        return {"status": "error", "message": str(e)}


def get_prediction(raw_input_data):
    """
    兼容性包装器：确保在调用预测前模型组件已加载。
    返回与原 `predict` 相同的字典结构。
    """
    # 如果模型或组件尚未加载，尝试加载一次
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        loaded = load_model_components()
        if not loaded:
            return {"status": "error", "message": "Model components not loaded and failed to load."}

    return predict(raw_input_data)

def build_attack_sample_library():
    """
    从原始数据集中，为每种攻击类型采样最多 5 条，构建攻击样本库。
    只使用 FEATURE_COLUMNS 中定义的特征，保证与模型输入一致。
    """
    global ATTACK_SAMPLE_LIBRARY

    with attack_samples_lock:
        # 已经构建过就直接返回
        if ATTACK_SAMPLE_LIBRARY:
            return True

        # 确保模型组件已加载，从而拿到 FEATURE_COLUMNS
        if not FEATURE_COLUMNS:
            loaded = load_model_components()
            if not loaded or not FEATURE_COLUMNS:
                logger.error("Cannot build attack sample library: FEATURE_COLUMNS not available.")
                return False

        # 检查数据集路径
        if not os.path.exists(ATTACK_DATASET_PATH):
            logger.error(f"Attack dataset file not found: {ATTACK_DATASET_PATH}")
            return False

        try:
            logger.info(f"Loading attack dataset from {ATTACK_DATASET_PATH} ...")
            df = pd.read_csv(ATTACK_DATASET_PATH)

            # 列名去空格，基础清洗
            df.columns = df.columns.str.strip()
            if 'Label' not in df.columns:
                logger.error("Dataset has no 'Label' column.")
                return False

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # 过滤出攻击行（排除 BENIGN）
            df['Label'] = df['Label'].astype(str).str.strip()
            attack_df = df[df['Label'].str.upper() != 'BENIGN']

            if attack_df.empty:
                logger.error("No attack rows found in dataset.")
                return False

            library = []

            # 按攻击类型分组，每类最多 5 条
            for label, group in attack_df.groupby('Label'):
                sample_n = min(5, len(group))
                sampled = group.sample(n=sample_n, random_state=42)

                for _, row in sampled.iterrows():
                    features = []
                    for col in FEATURE_COLUMNS:
                        if col in sampled.columns:
                            val = row[col]
                            try:
                                val = float(val)
                            except Exception:
                                val = 0.0
                        else:
                            val = 0.0
                        features.append(val)

                    library.append({
                        "label": label,
                        "features": features
                    })

            ATTACK_SAMPLE_LIBRARY = library
            logger.info(
                f"Attack sample library built: {len(ATTACK_SAMPLE_LIBRARY)} samples "
                f"from {attack_df['Label'].nunique()} attack types."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build attack sample library: {e}")
            return False

# ----------------------------------------------------------------------
# 4. 模型重训练逻辑
# ----------------------------------------------------------------------
def train_model_with_data(df, target_column='Label'):
    """
    使用上传的数据重新训练模型。

    返回值:
        字典 {'success': True/False, 'message': str, 'stats': dict}
        stats 包含: total_samples, label_distribution, new_labels_count
    """
    global PERFORMANCE_METRICS

    try:
        logger.info("Starting retraining process...")

        # 1. 简单清理
        df.columns = df.columns.str.strip()  # 去除列名空格

        # 检查是否存在 Label 列
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'stats': {}}

        # 处理 Inf 和 NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)  # 这里选择直接丢弃，保证训练质量

        # 2. 标签编码
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

        # 收集统计信息
        unique_labels = set(le.classes_)
        label_dist = df[target_column].value_counts().to_dict()
        old_labels = set(LE.classes_) if LE else set()
        new_labels = unique_labels - old_labels
        stats = {
            'total_samples': len(df),
            'label_distribution': {le.inverse_transform([k])[0]: v for k, v in label_dist.items()},
            'unique_labels': list(unique_labels),
            'new_labels': list(new_labels),
            'new_labels_count': len(new_labels)
        }
        logger.info(f"Data statistics: {stats}")

        # 3. 分离特征和标签
        X = df.drop(columns=[target_column])
        y = df[target_column]

        feature_columns_list = X.columns.tolist()

        # 4. 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 转换回 DataFrame 格式以保持一致性
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns_list)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns_list)

        # 6. 训练 Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # 7. 评估
        y_pred = rf_model.predict(X_test_scaled)

        # ====== AUC 计算（修复二分类越界问题）======
        # 1) 获取模型对测试集的概率输出 (AUC 必需)
        #    ✅ 必须和训练一致：用 X_test_scaled，而不是 X_test
        y_scores = rf_model.predict_proba(X_test_scaled)

        # 2) 类别
        classes = np.unique(y_test)
        n_classes = len(classes)

        # 3) 计算 AUC（保持你“加权 AUC”的思想）
        if n_classes == 2:
            # 二分类：label_binarize 只返回 (n,1)，不能 [:, i] 循环
            y_bin = label_binarize(y_test, classes=classes).ravel()  # (n,)

            # predict_proba 二分类一般是 (n,2)，取正类(classes[1])那列
            if y_scores.ndim == 2 and y_scores.shape[1] >= 2:
                score_pos = y_scores[:, 1]
            else:
                score_pos = np.asarray(y_scores).ravel()

            fpr, tpr, _ = roc_curve(y_bin, score_pos)
            auc_weighted = float(auc(fpr, tpr))

        else:
            # 多分类：OvR，每类算 AUC，再按支持度加权平均
            y_test_binarized = label_binarize(y_test, classes=classes)  # (n, n_classes)

            support = y_test.value_counts().sort_index().values
            total_support = np.sum(support)

            roc_auc = dict()
            weighted_auc_sum = 0.0

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr, tpr)

                weight = support[i] / total_support
                weighted_auc_sum += roc_auc[i] * weight

            auc_weighted = float(weighted_auc_sum)
        # ====== AUC 计算结束 ======

        PERFORMANCE_METRICS = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "FPR": 1.0 - recall_score(y_test, y_pred, average=None, zero_division=0)[0],
            "auc": auc_weighted
        }

        # 8. 保存所有组件 (覆盖旧文件)
        os.makedirs('./models', exist_ok=True)
        joblib.dump(rf_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le, ENCODER_PATH)
        joblib.dump(feature_columns_list, FEATURE_COLS_PATH)

        logger.info(f"Retraining complete. Accuracy: {PERFORMANCE_METRICS['accuracy']:.4f}")

        # 保存性能指标
        with open(PERFORMANCE_PATH, 'w') as f:
            json.dump(PERFORMANCE_METRICS, f)
        logger.info(f"Performance metrics saved to {PERFORMANCE_PATH}")

        return {
            'success': True,
            'message': f'Retraining complete. Accuracy: {PERFORMANCE_METRICS["accuracy"]:.4f}',
            'stats': stats
        }

    except Exception as e:
        logger.error(f"Train model with data failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e), 'stats': {}}

# ----------------------------------------------------------------------
# 5. API 路由接口
# ----------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    预测接口
    POST Body: {"features": [v1, v2, ...]}
    """
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        return jsonify({"status": "error", "message": "Model not fully loaded."}), 503

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"status": "error", "message": "Missing 'features' in JSON."}), 400

        features = data['features']

        # 调用预测
        result = predict(features)

        if result['status'] == 'success':
            # 保存到数据库
            try:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO detection_history (timestamp, predicted_label, confidence, threat_level, features_count) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     result['predicted_label'],
                     result['confidence'],
                     result['threat_level'],
                     len(features)))
                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Database insert error: {db_e}")

            # 处理警报 (非正常流量)
            if result['predicted_label'].upper() != 'BENIGN':
                with alerts_lock:
                    alert = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": result['predicted_label'],
                        "confidence": result['confidence'],
                        "level": result['threat_level']
                    }
                    alerts.append(alert)
                    # 修复：使用全局变量 MAX_ALERTS
                    if len(alerts) > MAX_ALERTS:
                        alerts.pop(0)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """获取最新警报"""
    with alerts_lock:
        return jsonify(list(reversed(alerts)))


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, predicted_label, confidence, threat_level FROM detection_history ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "type": row[1],
                "confidence": row[2],
                "level": row[3],  # 直接使用 threat_level
                "threat_level": row[3]
            })
        return jsonify(history)
    except Exception as e:
        return jsonify([]), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """获取模型性能"""
    return jsonify(PERFORMANCE_METRICS)
import random
import time
from flask import request, jsonify

# 固定“攻击间隔”
MODE_INTERVAL_MS = {
    "Low": 2000,     # 2秒一次
    "Medium": 1000,  # 1秒一次
    "High": 500,     # 0.5秒一次（=1秒两次）
}

# 前端根据“最近10秒次数”分级用的阈值
# 10秒内：Low=5次，Medium=10次，High=20次
LEVEL_THRESHOLDS = {
    "low_max": 5,      # 0~5 => Low
    "medium_max": 10,  # 6~10 => Medium
    # >=11 => High
}

def count_to_level(c: int) -> str:
    if c <= LEVEL_THRESHOLDS["low_max"]:
        return "Low"
    if c <= LEVEL_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"

# ====== 放在 app.py 顶部 import 附近（如果已有就不用重复）======
import os
import random
import time
import pandas as pd
from threading import Lock
from flask import request, jsonify

# ====== 异常流量 CSV 缓存（新增）======
ANOMALY_TRAFFIC_DF = None
ANOMALY_FEATURE_COLS = None
ANOMALY_LABEL_COL = None
anomaly_traffic_lock = Lock()

def _resolve_anomaly_csv_path() -> str:
    """
    兼容不同启动目录：优先按 app.py 所在目录定位 stream/anomaly_traffic.csv
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "stream", "anomaly_traffic.csv"),
        os.path.join(os.getcwd(), "stream", "anomaly_traffic.csv"),
        "stream/anomaly_traffic.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # 默认返回第一个（用于报错提示）

def _load_anomaly_traffic():
    """
    只加载一次 anomaly_traffic.csv，并缓存：
      - ANOMALY_TRAFFIC_DF
      - ANOMALY_FEATURE_COLS
      - ANOMALY_LABEL_COL（如果能识别）
    """
    global ANOMALY_TRAFFIC_DF, ANOMALY_FEATURE_COLS, ANOMALY_LABEL_COL

    if ANOMALY_TRAFFIC_DF is not None and ANOMALY_FEATURE_COLS is not None:
        return

    with anomaly_traffic_lock:
        if ANOMALY_TRAFFIC_DF is not None and ANOMALY_FEATURE_COLS is not None:
            return

        csv_path = _resolve_anomaly_csv_path()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"anomaly traffic csv not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # 1) 尝试识别 label 列（可选）
        label_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("label", "class", "target", "y"):
                label_col = c
                break

        # 2) 选择特征列：优先用你模型的 FEATURE_COLUMNS（如果存在且匹配）
        feature_cols = None
        try:
            if "FEATURE_COLUMNS" in globals() and FEATURE_COLUMNS:
                # FEATURE_COLUMNS 可能是 list[str]
                if all(col in df.columns for col in FEATURE_COLUMNS):
                    feature_cols = list(FEATURE_COLUMNS)
        except Exception:
            pass

        # 3) 如果 CSV 不包含完整 FEATURE_COLUMNS，则退化为：取所有数值列（排除 label）
        if feature_cols is None:
            tmp = df.copy()
            if label_col and label_col in tmp.columns:
                tmp = tmp.drop(columns=[label_col])
            # 只保留数值列
            tmp = tmp.select_dtypes(include=["number"])
            feature_cols = list(tmp.columns)

        if not feature_cols:
            raise ValueError("No numeric feature columns found in anomaly_traffic.csv")

        # 4) 如果模型期望 78（或 FEATURE_COLUMNS 长度），这里做一致性校验
        expected = None
        if "FEATURE_COLUMNS" in globals() and FEATURE_COLUMNS:
            expected = len(FEATURE_COLUMNS)
        # 没有 FEATURE_COLUMNS 就不强制，但一般你模型是 78
        if expected is not None and len(feature_cols) != expected:
            raise ValueError(
                f"Feature count mismatch: csv has {len(feature_cols)} cols, "
                f"model expects {expected}. Check anomaly_traffic.csv columns."
            )

        ANOMALY_TRAFFIC_DF = df
        ANOMALY_FEATURE_COLS = feature_cols
        ANOMALY_LABEL_COL = label_col


# ====== 保持你之前的“三档固定间隔 + 阈值逻辑不变”======
MODE_INTERVAL_MS = {
    "Low": 2000,     # 2秒一次
    "Medium": 1000,  # 1秒一次
    "High": 500,     # 0.5秒一次（1秒两次）
}

LEVEL_THRESHOLDS = {
    "low_max": 5,      # 0~5 => Low
    "medium_max": 10,  # 6~10 => Medium
    # >=11 => High
}

def count_to_level(c: int) -> str:
    if c <= LEVEL_THRESHOLDS["low_max"]:
        return "Low"
    if c <= LEVEL_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


# ====== ✅ 替换原来的 /api/stream 路由为下面这个 ======
@app.route('/api/stream', methods=['GET'])
def get_attack_stream_sample():
    """
    改动点：异常流量从 stream/anomaly_traffic.csv 随机抽取（每次事件随机一行）
    其他逻辑保持：
      - mode 三档随机（可用 ?mode= 指定）
      - Low: 2s一次, Medium: 1s一次, High: 0.5s一次
      - window 默认 TIME_WINDOW_SECONDS（默认10）
      - 返回 stream[{features,label,at_ms,ts}]
    """
    try:
        # 1) 加载异常流量 CSV（只加载一次）
        _load_anomaly_traffic()

        df = ANOMALY_TRAFFIC_DF
        feature_cols = ANOMALY_FEATURE_COLS
        label_col = ANOMALY_LABEL_COL

        # 2) window 秒数：默认 TIME_WINDOW_SECONDS，没有就 10
        default_window = int(TIME_WINDOW_SECONDS) if "TIME_WINDOW_SECONDS" in globals() else 10
        window_s = int(request.args.get("window", default_window))
        window_ms = max(1000, window_s * 1000)

        # 3) mode：不传则随机三选一
        mode = request.args.get("mode")
        if mode not in MODE_INTERVAL_MS:
            mode = random.choice(list(MODE_INTERVAL_MS.keys()))

        interval = MODE_INTERVAL_MS[mode]

        # 4) label 过滤（如果 CSV 有 label 列才支持）
        label_filter = request.args.get("label")
        df_candidates = df
        if label_filter:
            if not label_col:
                return jsonify({
                    "status": "error",
                    "message": "label filter requested but anomaly_traffic.csv has no label column."
                }), 400
            df_candidates = df[df[label_col] == label_filter]
            if df_candidates.empty:
                return jsonify({
                    "status": "error",
                    "message": f"No samples found for label '{label_filter}'."
                }), 404

        # 5) 按固定间隔生成 at_ms（保持原逻辑：10秒=>Low=5, Med=10, High=20）
        offsets = list(range(0, window_ms, interval))
        attack_frequency = len(offsets)

        # 6) 每个事件随机抽一行异常流量作为 features
        start_ms = int(time.time() * 1000)
        stream = []
        n = len(df_candidates)

        for off in offsets:
            # 随机行
            ridx = random.randrange(n)
            row = df_candidates.iloc[ridx]

            # features：转成 float list，NaN 用 0.0
            feats = pd.to_numeric(row[feature_cols], errors="coerce").fillna(0.0).astype(float).tolist()

            # label：优先用 CSV 自带 label；否则 fallback
            lbl = None
            if label_col:
                lbl = row[label_col]
            elif label_filter:
                lbl = label_filter
            else:
                lbl = "ANOMALY"

            stream.append({
                "features": feats,
                "label": str(lbl) if lbl is not None else "ANOMALY",
                "at_ms": off,
                "ts": start_ms + off
            })

        return jsonify({
            "status": "ok",
            "mode": mode,
            "time_window_seconds": window_s,
            "attack_frequency": attack_frequency,
            "frequency_level": count_to_level(attack_frequency),
            "thresholds": LEVEL_THRESHOLDS,
            "stream": stream
        })

    except Exception as e:
        logger.error(f"/api/stream error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/random', methods=['GET'])
def get_random_data():
    """生成随机数据"""
    count = len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 78
    data = np.random.uniform(0, 1000, count).tolist()
    names = FEATURE_COLUMNS if FEATURE_COLUMNS else [f"f_{i}" for i in range(count)]
    return jsonify({"features": data, "feature_names": names})


@app.route('/api/upload-and-retrain', methods=['POST'])
def upload_and_retrain():
    """
    上传 CSV 并重训练
    """
    try:
        if 'files' not in request.files:
            return jsonify({"status": "error", "message": "No files part"}), 400

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"status": "error", "message": "No selected file"}), 400

        dfs = []
        for file in files:
            if file and file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Error reading {file.filename}: {e}"}), 400

        if not dfs:
            return jsonify({"status": "error", "message": "No valid CSV files found"}), 400

        full_df = pd.concat(dfs, ignore_index=True)

        # 启动训练
        result = train_model_with_data(full_df)

        if result['success']:
            # 重新加载模型
            if load_model_components():
                return jsonify({
                    "status": "success",
                    "message": result['message'],
                    "stats": result['stats'],
                    "performance": PERFORMANCE_METRICS
                })
            else:
                return jsonify({"status": "error", "message": "Training succeeded but reload failed."}), 500
        else:
            return jsonify({"status": "error", "message": result['message'], "details": result['stats']}), 400

    except Exception as e:
        logger.error(f"Retrain API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------------------------------------------
# 程序入口
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 启动时加载模型
    if not load_model_components():
        logger.warning("⚠️ Warning: Model components could not be loaded at startup.")
        logger.warning("   Please ensure 'trainning.py' has been run and generated files in './models/.")

    app.run(host='127.0.0.1', port=5050, debug=True)