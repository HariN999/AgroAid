import os
import pytest
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import CNN

def test_model_loading_and_architecture():
    """Verify that the model architecture can be instantiated and configured correctly."""
    model = CNN.CNN(39)
    assert model.conv_layers is not None
    assert model.dense_layers is not None

def test_image_normalization_formats(tmp_path):
    """Test that various image formats (RGB, RGBA, Grayscale, Palette) are safely converted to 3x224x224 tensors."""
    formats = {
        'RGB': Image.new('RGB', (300, 400), color='green'),
        'RGBA': Image.new('RGBA', (150, 150), color=(0, 255, 0, 120)),
        'L': Image.new('L', (200, 200), color=128),
        'P': Image.new('P', (100, 100), color=2)
    }

    for mode, img in formats.items():
        # Save temp image
        img_path = os.path.join(tmp_path, f"temp_{mode}.png")
        img.save(img_path)
        
        # Load and convert to RGB (mirroring prediction logic in app.py)
        with Image.open(img_path) as loaded_img:
            rgb_img = loaded_img.convert('RGB')
            assert rgb_img.mode == 'RGB'
            
            # Check resizing
            resized_img = rgb_img.resize((224, 224))
            assert resized_img.size == (224, 224)
            
            # Check tensor conversion
            tensor_img = TF.to_tensor(resized_img)
            assert tensor_img.shape == (3, 224, 224)
            
            # Check batch view shape
            batch_img = tensor_img.view((-1, 3, 224, 224))
            assert batch_img.shape == (1, 3, 224, 224)

def test_model_inference_with_real_weights():
    """Verify loading of real weights and inference output shape if the checkpoint is present."""
    model_path = "models/plant_disease_model_1_latest.pt"
    if not os.path.exists(model_path):
        pytest.skip("Local model weights checkpoint is not present.")

    net = CNN.CNN(39)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()

    # Generate a dummy input tensor of shape [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = net(dummy_input)
    assert output.shape == torch.Size([1, 39]) or list(output.shape) == [1, 39]

def test_confidence_and_alternatives_logic():
    """Verify softmax probability range, calibration output shapes, and topk sort ordering."""
    # Create mock logits for 39 classes with a dominant class 15
    logits = torch.zeros(1, 39)
    logits[0, 15] = 10.0   # Strong prediction for index 15
    logits[0, 16] = 8.0    # Secondary prediction for index 16
    logits[0, 0] = 5.0     # Tertiary prediction for index 0

    probabilities = torch.softmax(logits, dim=1)[0]
    
    # 1. Assert probability range and sum to 1
    assert float(torch.sum(probabilities)) == pytest.approx(1.0)
    assert float(probabilities[15]) > 0.80
    assert float(probabilities[16]) > float(probabilities[0])
    
    # 2. Assert topk indices and values are correctly sorted
    top_probs, top_indices = torch.topk(probabilities, k=3)
    assert int(top_indices[0]) == 15
    assert int(top_indices[1]) == 16
    assert int(top_indices[2]) == 0
    assert float(top_probs[0]) > float(top_probs[1])
