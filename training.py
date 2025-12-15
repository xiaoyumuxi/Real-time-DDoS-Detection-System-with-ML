import joblib
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
# File path configuration
file_path = './data/Wednesday-workingHours.pcap_ISCX.csv'

print("Loading data...")

# 1. Read CSV file
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully, shape: {df.shape}")
except FileNotFoundError:
    print("Error: File not found, please check the file path.")
    exit()

# Data preprocessing
df.columns = df.columns.str.strip()

# 2. Data cleaning
print("Cleaning data...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"Shape after cleaning: {df.shape}")

# Multi-class label encoding
print("Encoding labels...")

print("Original label categories:", df['Label'].unique())

# 3. Label encoding
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel mapping:")
for label, num in label_mapping.items():
    print(f"  {label} -> {num}")

print("\nEncoded label distribution:")
print(df['Label'].value_counts())

# 4. Separate features and labels
y = df['Label']
X = df.drop('Label', axis=1)

FEATURE_COLUMNS = X.columns.tolist()

# 5. Split train/test sets
print("\nSplitting train/test sets (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Feature scaling
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nPreprocessing complete!")
print("Feature standardization complete.")

# 5. Model training
print("\n--- Step 5: Training Random Forest classifier ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Model training complete.")

# 6. Model evaluation
print("\n--- Step 6: Model evaluation ---")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
tnr_score = recalls[0]
FPR = 1.0 - tnr_score
y_scores = rf_model.predict_proba(X_test)


classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)


support = y_test.value_counts().sort_index().values
total_support = np.sum(support)

roc_auc = dict()
weighted_auc_sum = 0

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr, tpr)
    weight = support[i] / total_support
    weighted_auc_sum += roc_auc[i] * weight

auc_weighted = weighted_auc_sum


print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print("---------------------------------")
print(f"Test Recall [BENIGN]: {recall:.4f}")
print(f"False Positive Rate: {FPR:.4f}")
print(f"Test AUC: {auc_weighted:.4f}")
performance_metrics = {
    "accuracy": f"{accuracy:.4f}",
    "precision": f"{precision:.4f}",
    "recall": f"{recall:.4f}",
    "FPR": f"{FPR:.4f}",
    "auc": f"{auc_weighted:.4f}"
}

metrics_filename = './models/ddos_performance.json'
with open(metrics_filename, 'w') as f:
    json.dump(performance_metrics, f)

print(f"- Performance metrics: {metrics_filename}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# 7. Save model and preprocessors
print("\n--- Step 7: Saving model and preprocessors ---")

model_filename = './models/ddos_rf_model.joblib'
scaler_filename = './models/ddos_scaler.joblib'
encoder_filename = './models/ddos_label_encoder.joblib'
feature_col = './models/ddos_feature_columns.joblib'


joblib.dump(rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(le, encoder_filename)
joblib.dump(FEATURE_COLUMNS, feature_col)

print(f" Task complete!")
print(f"Model and preprocessors saved as:\n- Model: {model_filename}\n- Scaler: {scaler_filename}\n- Encoder: {encoder_filename}\n- Features: {feature_col}")