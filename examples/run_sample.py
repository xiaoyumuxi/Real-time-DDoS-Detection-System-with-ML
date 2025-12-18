import json
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app import get_prediction

# Note: Sample data must match model's FEATURE_COLUMNS length
SAMPLE_RAW_DATA = [
    54865, 3, 2, 0, 12, 0, 6, 6, 6.0, 0.0, 0, 0, 0.0, 0.0, 4000000.0,
    666666.6667, 3.0, 0.0, 3, 3, 3, 3.0, 0.0, 3, 3, 0, 0.0, 0.0, 0, 0,
    0, 0, 0, 0, 40, 0, 666666.6667, 0.0, 6, 6, 6.0, 0.0, 0.0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 9.0, 6.0, 0.0, 40, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0,
    33, -1, 1, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
]

if __name__ == '__main__':
    result = get_prediction(SAMPLE_RAW_DATA)
    print("\n--- API Response Result ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    if result.get('status') == 'success':
        if result.get('predicted_label', '').upper() == 'BENIGN':
            print("\n🟢 Prediction: Normal traffic (BENIGN)")
        else:
            print(f"\n🔴 Prediction: Malicious attack detected! Type: {result.get('predicted_label')}")
            attack_types = {
                'DDOS': 'Distributed Denial of Service',
                'DOS': 'Denial of Service',
                'PORTSCAN': 'Port Scan Attack',
                'BOT': 'Botnet Activity',
                'INFLITRATION': 'Infiltration Attack',
                'BRUTEFORCE': 'Brute Force Attack',
                'SQLINJECTION': 'SQL Injection',
                'XSS': 'Cross-Site Scripting',
                'FTP-PATATOR': 'FTP Password Brute Force',
                'SSH-PATATOR': 'SSH Password Brute Force'
            }
            attack_description = attack_types.get(result.get('predicted_label','').upper(), 'Unknown attack type')
            print(f"Attack description: {attack_description}")
