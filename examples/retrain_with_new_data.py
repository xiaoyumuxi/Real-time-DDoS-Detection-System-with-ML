#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""Retraining Example Script

This script demonstrates how to:
1. Read CSV file from local disk
2. Upload to Flask backend's /api/upload-and-retrain endpoint
3. Trigger model retraining
4. Get new model performance metrics and statistics

Usage:
    python examples/retrain_with_new_data.py <csv_file_path> [--host HOST] [--port PORT]

Examples:
    python examples/retrain_with_new_data.py ./data/Wednesday-workingHours.pcap_ISCX.csv
    python examples/retrain_with_new_data.py ./data/custom_attack_data.csv --host 127.0.0.1 --port 5000
"""

import requests
import argparse
import sys
from pathlib import Path


def retrain_model(csv_file, host='127.0.0.1', port=5000):
    """
    Send CSV file to backend and trigger retraining
    
    Args:
        csv_file (str): Local path to CSV file
        host (str): Flask server host
        port (int): Flask server port
    
    Returns:
        dict: Response data
    """
    csv_path = Path(csv_file)
    
    # Validate file exists
    if not csv_path.exists():
        print(f"Error: File not found {csv_file}")
        return None
    
    if not csv_file.endswith('.csv'):
        print(f"Error: File must be CSV format {csv_file}")
        return None
    
    # Build API URL
    url = f'http://{host}:{port}/api/upload-and-retrain'
    
    print(f"Uploading file: {csv_file}")
    print(f"API endpoint: {url}")
    
    try:
        # Open file and send POST request
        with open(csv_file, 'rb') as f:
            files = {'files': f}
            response = requests.post(url, files=files, timeout=300)
        
        # Process response
        if response.status_code == 200:
            data = response.json()
            print(f"\nRetraining successful!")
            print(f"Status: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            
            # Display statistics
            if 'stats' in data:
                stats = data['stats']
                print(f"\nData statistics:")
                print(f"  • Total samples: {stats.get('total_samples', 'N/A')}")
                print(f"  • Unique labels: {len(stats.get('unique_labels', []))}")
                print(f"  • New labels: {stats.get('new_labels_count', 0)}"))
                if stats.get('new_labels'):
                    print(f"  • New labels added: {', '.join(stats['new_labels'])}")
                
                print(f"\nLabel distribution:")
                for label, count in stats.get('label_distribution', {}).items():
                    print(f"  • {label}: {count}")
            
            # Display performance metrics
            if 'performance' in data:
                perf = data['performance']
                print(f"\nModel performance metrics:")
                print(f"  • Accuracy: {perf.get('accuracy', 'N/A'):.4f}")
                print(f"  • Precision: {perf.get('precision', 'N/A'):.4f}")
                print(f"  • Recall: {perf.get('recall', 'N/A'):.4f}")
                print(f"  • F1-Score: {perf.get('f1_score', 'N/A'):.4f}"))
            
            return data
        
        else:
            print(f"\nRequest failed, status code: {response.status_code}"))
            try:
                error_data = response.json()
                print(f"Error message: {error_data.get('message', 'Unknown error')}")
                if 'details' in error_data:
                    print(f"Details: {error_data['details']}")
            except:
                print(f"Response: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"\nConnection error: Cannot connect to {url}")
        print(f"   Please ensure Flask server is running (python app.py)"))
        return None
    
    except requests.exceptions.Timeout:
        print(f"\nTimeout: Request took too long (300s)")
        print(f"   CSV file may be too large, try using a smaller file"))
        return None
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}"))
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Upload CSV file and retrain DDoS detection model',
        epilog='Example: python retrain_with_new_data.py ./data/custom_data.csv --host 127.0.0.1 --port 5000'
    )
    
    parser.add_argument(
        'csv_file',
        help='CSV file path (must contain Label column)'
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Flask server host (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Flask server port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Call retraining function
    result = retrain_model(args.csv_file, args.host, args.port)
    
    # Return exit code
    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
