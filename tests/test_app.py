import os
import io
import pytest
from unittest.mock import patch
from PIL import Image
from app import app, Config

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_static_routes(client):
    """Test that all basic static pages load successfully."""
    routes = ['/', '/index']
    for r in routes:
        response = client.get(r)
        assert response.status_code == 200

def test_submit_missing_file(client):
    """Test that submitting without file field returns 400."""
    response = client.post('/submit', data={})
    assert response.status_code == 400
    assert b"No image uploaded" in response.data

def test_submit_empty_filename(client):
    """Test that submitting an empty filename field returns 400."""
    data = {
        'image': (io.BytesIO(b""), '')
    }
    response = client.post('/submit', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b"No file selected" in response.data

def test_submit_unsupported_file_type(client):
    """Test that submitting an unsupported file extension returns 400."""
    data = {
        'image': (io.BytesIO(b"content"), 'test.txt')
    }
    response = client.post('/submit', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b"Unsupported file type" in response.data

def test_submit_corrupted_image(client):
    """Test that uploading a corrupted/invalid image returns 400."""
    data = {
        'image': (io.BytesIO(b"corrupted raw data"), 'test.jpg')
    }
    response = client.post('/submit', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b"Invalid or corrupted image file" in response.data

@patch('app.prediction')
def test_submit_valid_image_and_cleanup(mock_predict, client):
    """Test that a valid image upload runs prediction, renders result page, and cleans up uploaded files."""
    mock_predict.return_value = (0, 0.95, [])  # Apple scab index, 95% confidence, no alternatives
    
    # Generate a valid tiny JPEG image in memory
    img_byte_arr = io.BytesIO()
    img = Image.new('RGB', (50, 50), color='green')
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {
        'image': (img_byte_arr, 'leaf.jpg')
    }
    
    # Verify uploads directory starts empty or clean
    existing_files_before = set(os.listdir(Config.UPLOAD_FOLDER))
    
    response = client.post('/submit', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Apple : Scab" in response.data
    
    # Verify temporary uploaded file was cleaned up and deleted
    existing_files_after = set(os.listdir(Config.UPLOAD_FOLDER))
    # Exclude .gitkeep or other pre-existing files
    difference = (existing_files_after - existing_files_before) - {'.gitkeep'}
    assert len(difference) == 0

def test_translate_valid_index(client):
    """Test standard stateless translation request with prediction index."""
    response = client.get('/translate?lang=en&pred=0')
    assert response.status_code == 200
    data = response.get_json()
    assert data['title'] == "Apple : Scab"
    assert 'desc' in data
    assert 'prevent' in data
    assert 'active_ingredient' in data
    assert 'treatment_type' in data
    assert 'purpose' in data

def test_translate_invalid_indexes(client):
    """Test translation bounds validation."""
    # Index out of bounds
    response1 = client.get('/translate?lang=en&pred=99')
    assert response1.status_code == 404
    assert b"Disease details not found in dataset" in response1.data

    # Missing index
    response2 = client.get('/translate?lang=en')
    assert response2.status_code == 400
    assert b"No disease index provided" in response2.data

    # Non-integer index
    response3 = client.get('/translate?lang=en&pred=abc')
    assert response3.status_code == 400
    assert b"Invalid disease index" in response3.data

@patch('app.translate_text')
def test_translate_api_failure_fallback(mock_translate_text, client):
    """Test that if the external translation API fails, the server falls back gracefully to English."""
    mock_translate_text.side_effect = Exception("Translation API timed out")
    
    response = client.get('/translate?lang=hi&pred=0')
    assert response.status_code == 200  # Fallback: must not return 500
    data = response.get_json()
    assert data['title'] == "Apple : Scab"  # Original English content

def test_treatment_options_behavior(client):
    """Test treatment model lookup, alignment, and healthy crop checks."""
    # 1. Test index-safe verified treatment options lookup for Apple Scab (index 0)
    response = client.get('/translate?lang=en&pred=0')
    assert response.status_code == 200
    data = response.get_json()
    assert data['active_ingredient'] == "Propiconazole"
    assert data['treatment_type'] == "Systemic Fungicide"
    assert data['purpose'] == "Scab Management"
    assert data['verification_status'] == "Verified"

    # 2. Test unverified treatment options lookup for Apple Black Rot (index 1)
    response_unverified = client.get('/translate?lang=en&pred=1')
    assert response_unverified.status_code == 200
    data_unverified = response_unverified.get_json()
    assert data_unverified['active_ingredient'] == "Not established"
    assert data_unverified['verification_status'] == "Needs verification"

    # 3. Test healthy class checks (index 3 is Apple healthy) - must not recommend chemical pesticide
    response_healthy = client.get('/translate?lang=en&pred=3')
    assert response_healthy.status_code == 200
    data_healthy = response_healthy.get_json()
    assert data_healthy['active_ingredient'] == "None"
    assert data_healthy['treatment_type'] == "None"
    assert data_healthy['purpose'] == "Healthy maintenance"
    assert data_healthy['verification_status'] == "Verified"
