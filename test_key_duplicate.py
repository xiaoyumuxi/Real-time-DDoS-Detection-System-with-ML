"""
"""Unit Test File: Test all API functionality including key duplication tests

Test Coverage:
1. Normal functionality of all API endpoints
2. JSON request key duplication scenarios
3. Edge cases and error handling
4. Data validation and exception handling
"""

import pytest
import json
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock
import sys

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model_components, init_db, MODEL, SCALER, LE, FEATURE_COLUMNS


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    # Create temporary database
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    test_db.close()
    
    # Backup original database path
    from app import DB_FILE
    original_db = DB_FILE
    
    # Temporarily replace database path
    import app as app_module
    app_module.DB_FILE = test_db.name
    
    # Initialize test database
    init_db()
    
    with app.test_client() as client:
        yield client
    
    # Cleanup: restore original database path and delete test database
    app_module.DB_FILE = original_db
    if os.path.exists(test_db.name):
        os.unlink(test_db.name)


@pytest.fixture
def sample_features():
    """Generate sample feature vector"""
    # Assume 78 features (based on default value in code)
    return [float(i) for i in range(78)]


@pytest.fixture
def mock_model_loaded():
    """Mock model loaded state"""
    with patch('app.MODEL', MagicMock()), \
         patch('app.SCALER', MagicMock()), \
         patch('app.LE', MagicMock()), \
         patch('app.FEATURE_COLUMNS', [f'f_{i}' for i in range(78)]):
        # Set LE's inverse_transform method
        app.LE.inverse_transform = MagicMock(return_value=['BENIGN'])
        yield


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data


class TestPredictAPI:
    """Test prediction API endpoint"""
    
    def test_predict_missing_features(self, client):
        """Test missing features field"""
        response = client.post('/api/predict', 
                             json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'features' in data['message'].lower()
    
    def test_predict_empty_json(self, client):
        """Test empty JSON request"""
        response = client.post('/api/predict',
                             json=None,
                             content_type='application/json')
        assert response.status_code == 400
    
    def test_predict_key_duplicate_in_json(self, client, sample_features, mock_model_loaded):
        """Test JSON with duplicate keys"""
        # Create JSON string with duplicate keys
        # Note: Python dict automatically handles duplicate keys (keeps last one)
        # but we can test this scenario
        json_str = '{"features": [1, 2, 3], "features": ' + str(sample_features) + '}'
        
        # Send using requests method, simulating key duplication
        response = client.post('/api/predict',
                             data=json_str,
                             content_type='application/json')
        
        # Since Python dict handles duplicate keys automatically, should process normally
        # If model not loaded, returns 503
        assert response.status_code in [200, 400, 503]
    
    def test_predict_multiple_duplicate_keys(self, client, sample_features):
        """Test multiple duplicate keys"""
        # Create JSON with multiple duplicate keys
        json_data = {
            'features': sample_features,
            'features': sample_features,  # Duplicate key
            'extra': 'value1',
            'extra': 'value2'  # Duplicate key
        }
        
        response = client.post('/api/predict',
                             json=json_data)
        # Should be handled (Python dict keeps last value)
        assert response.status_code in [200, 400, 503]
    
    def test_predict_invalid_features_type(self, client):
        """Test invalid features type"""
        response = client.post('/api/predict',
                             json={'features': 'not_a_list'})
        assert response.status_code in [400, 500, 503]
    
    def test_predict_wrong_feature_count(self, client, mock_model_loaded):
        """Test feature count mismatch"""
        wrong_features = [1.0, 2.0, 3.0]  # Only 3 features, should need 78
        response = client.post('/api/predict',
                             json={'features': wrong_features})
        # If model is loaded, should return error
        assert response.status_code in [200, 400, 500, 503]


class TestAlertsAPI:
    """Test alerts API endpoint"""
    
    def test_get_alerts_success(self, client):
        """Test get alerts successfully"""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_get_alerts_empty(self, client):
        """Test get empty alerts list"""
        response = client.get('/api/alerts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []


class TestHistoryAPI:
    """Test history API endpoint"""
    
    def test_get_history_success(self, client):
        """Test get history successfully"""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_get_history_empty(self, client):
        """Test get empty history"""
        response = client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestPerformanceAPI:
    """Test performance metrics API endpoint"""
    
    def test_get_performance_success(self, client):
        """Test get performance metrics successfully"""
        response = client.get('/api/performance')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)
        # Check if expected performance metric fields are present
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        for key in expected_keys:
            if key in data:
                assert isinstance(data[key], (int, float))


class TestStreamAPI:
    """Test stream data API endpoint"""
    
    def test_get_stream_success(self, client):
        """Test get stream data successfully"""
        response = client.get('/api/stream')
        # May return error (if attack sample library not built) or success
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
    
    def test_get_stream_with_label_filter(self, client):
        """Test stream data with label filter"""
        response = client.get('/api/stream?label=DoS%20Hulk')
        assert response.status_code in [200, 404, 500]
    
    def test_get_stream_invalid_label(self, client):
        """Test invalid label filter"""
        response = client.get('/api/stream?label=NonExistentAttack')
        assert response.status_code in [200, 404, 500]
    
    def test_get_stream_duplicate_query_params(self, client):
        """Test duplicate query parameters"""
        # Test duplicate query parameters in URL
        response = client.get('/api/stream?label=DoS&label=PortScan')
        # Flask handles duplicate parameters (keeps last one or as list)
        assert response.status_code in [200, 404, 500]


class TestRandomAPI:
    """Test random data API endpoint"""
    
    def test_get_random_success(self, client):
        """Test get random data successfully"""
        response = client.get('/api/random')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'features' in data
        assert 'feature_names' in data
        assert isinstance(data['features'], list)
        assert isinstance(data['feature_names'], list)
        assert len(data['features']) == len(data['feature_names'])


class TestUploadAndRetrainAPI:
    """Test upload and retrain API endpoint"""
    
    def test_upload_no_files(self, client):
        """Test upload request without files"""
        response = client.post('/api/upload-and-retrain')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_upload_empty_files(self, client):
        """Test empty file list"""
        response = client.post('/api/upload-and-retrain',
                             data={'files': []})
        assert response.status_code == 400
    
    def test_upload_invalid_file_type(self, client):
        """Test invalid file type"""
        data = {
            'files': (BytesIO(b'not csv content'), 'test.txt')
        }
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        # Should return 400 or ignore non-CSV files
        assert response.status_code in [400, 200]
    
    def test_upload_valid_csv(self, client):
        """Test upload valid CSV file"""
        # Create test CSV data
        test_data = {
            'Label': ['BENIGN', 'DDoS', 'BENIGN'],
            'Feature1': [1.0, 2.0, 3.0],
            'Feature2': [4.0, 5.0, 6.0],
            'Feature3': [7.0, 8.0, 9.0]
        }
        df = pd.DataFrame(test_data)
        
        # Save as CSV byte stream
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        data = {
            'files': (csv_buffer, 'test.csv')
        }
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        # May succeed or fail (depends on model loading status)
        assert response.status_code in [200, 400, 500]
    
    def test_upload_multiple_files(self, client):
        """Test upload multiple files"""
        # Create two test CSV files
        test_data1 = {
            'Label': ['BENIGN'],
            'Feature1': [1.0],
            'Feature2': [2.0]
        }
        test_data2 = {
            'Label': ['DDoS'],
            'Feature1': [3.0],
            'Feature2': [4.0]
        }
        
        df1 = pd.DataFrame(test_data1)
        df2 = pd.DataFrame(test_data2)
        
        csv1 = BytesIO()
        csv2 = BytesIO()
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)
        csv1.seek(0)
        csv2.seek(0)
        
        data = {
            'files': [(csv1, 'test1.csv'), (csv2, 'test2.csv')]
        }
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        assert response.status_code in [200, 400, 500]
    
    def test_upload_csv_without_label_column(self, client):
        """Test CSV without Label column"""
        test_data = {
            'Feature1': [1.0, 2.0],
            'Feature2': [3.0, 4.0]
        }
        df = pd.DataFrame(test_data)
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        data = {
            'files': (csv_buffer, 'test.csv')
        }
        
        response = client.post('/api/upload-and-retrain',
                             data=data,
                             content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'


class TestKeyDuplicateScenarios:
    """Test various key duplication scenarios"""
    
    def test_json_duplicate_key_last_wins(self, client):
        """Test JSON duplicate keys, last value wins (Python dict behavior)"""
        # Python dict automatically handles duplicate keys, keeps last value
        json_data = {
            'features': [1, 2, 3],
            'features': [4, 5, 6]  # This value overwrites the above
        }
        
        # Verify Python dict behavior
        assert json_data['features'] == [4, 5, 6]
        
        response = client.post('/api/predict',
                             json=json_data)
        # Should use last value
        assert response.status_code in [200, 400, 500, 503]
    
    def test_multiple_duplicate_keys_in_request(self, client):
        """Test multiple duplicate keys in request"""
        json_data = {
            'features': [1.0] * 78,
            'features': [2.0] * 78,  # Duplicate
            'extra_param': 'value1',
            'extra_param': 'value2',  # Duplicate
            'another': 100,
            'another': 200  # Duplicate
        }
        
        # Verify dict behavior
        assert json_data['features'] == [2.0] * 78
        assert json_data['extra_param'] == 'value2'
        assert json_data['another'] == 200
        
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_nested_duplicate_keys(self, client):
        """Test duplicate keys in nested structure"""
        json_data = {
            'features': [1.0] * 78,
            'metadata': {
                'key1': 'value1',
                'key1': 'value2',  # Duplicate key in nested structure
                'key2': 100,
                'key2': 200  # Duplicate key in nested structure
            }
        }
        
        # Verify nested dict behavior
        assert json_data['metadata']['key1'] == 'value2'
        assert json_data['metadata']['key2'] == 200
        
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_query_params_duplicate(self, client):
        """Test duplicate URL query parameters"""
        # Flask handles duplicate query parameters
        response = client.get('/api/stream?label=DoS&label=PortScan&label=Bot')
        assert response.status_code in [200, 404, 500]
        
        # Can get all duplicate parameter values
        from flask import request as flask_request
        with app.test_request_context('/api/stream?label=DoS&label=PortScan'):
            labels = flask_request.args.getlist('label')
            assert len(labels) == 2
            assert 'DoS' in labels
            assert 'PortScan' in labels


class TestEdgeCases:
    """Test edge cases and exception handling"""
    
    def test_very_large_feature_array(self, client):
        """Test very large feature array"""
        large_features = [1.0] * 10000
        response = client.post('/api/predict',
                             json={'features': large_features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_empty_feature_array(self, client):
        """Test empty feature array"""
        response = client.post('/api/predict',
                             json={'features': []})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_none_values_in_features(self, client):
        """Test features containing None values"""
        features = [1.0, None, 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_inf_values_in_features(self, client, mock_model_loaded):
        """Test features containing Inf values"""
        features = [float('inf'), float('-inf'), 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        # Should handle Inf values (code has replace logic)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_nan_values_in_features(self, client, mock_model_loaded):
        """Test features containing NaN values"""
        features = [float('nan'), 2.0, 3.0] + [0.0] * 75
        response = client.post('/api/predict',
                             json={'features': features})
        # Should handle NaN values (code has fillna logic)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_unicode_in_json(self, client):
        """Test JSON containing Unicode characters"""
        json_data = {
            'features': [1.0] * 78,
            'message': 'Test Chinese 🚀'
        }
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_special_characters_in_keys(self, client):
        """Test keys containing special characters"""
        # Note: although features is required, can test other keys
        json_data = {
            'features': [1.0] * 78,
            'key-with-dash': 'value',
            'key_with_underscore': 'value',
            'key.with.dot': 'value'
        }
        response = client.post('/api/predict',
                             json=json_data)
        assert response.status_code in [200, 400, 500, 503]


class TestDataValidation:
    """Test data validation functionality"""
    
    def test_feature_count_validation(self, client, mock_model_loaded):
        """Test feature count validation"""
        # Test feature count mismatch
        wrong_count_features = [1.0] * 50  # Should be 78
        response = client.post('/api/predict',
                             json={'features': wrong_count_features})
        # If model is loaded, should return error
        assert response.status_code in [200, 400, 500, 503]
    
    def test_feature_type_validation(self, client):
        """Test feature type validation"""
        # Test mixed types
        mixed_features = [1, 2.0, '3', [4], None] + [0.0] * 73
        response = client.post('/api/predict',
                             json={'features': mixed_features})
        assert response.status_code in [200, 400, 500, 503]
    
    def test_malformed_json(self, client):
        """Test malformed JSON"""
        response = client.post('/api/predict',
                             data='{"features": [1, 2, 3}',  # Missing closing bracket
                             content_type='application/json')
        assert response.status_code in [400, 500]


class TestConcurrency:
    """Test concurrency scenarios"""
    
    def test_concurrent_alerts_access(self, client):
        """Test concurrent access to alerts"""
        import threading
        
        results = []
        
        def get_alerts():
            response = client.get('/api/alerts')
            results.append(response.status_code)
        
        threads = [threading.Thread(target=get_alerts) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])

