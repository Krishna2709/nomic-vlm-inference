"""
Unit tests for the ColPali API endpoints.

This module contains comprehensive tests for all API functionality including:
- Health endpoint
- Text embeddings (col and dense variants)
- Image embeddings
- Error handling
- Authentication
"""

import base64
import io
import json
import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from fastapi.testclient import TestClient
from PIL import Image
import torch

# Import the app and models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from app import app, TextInput, ImageInput, EmbedOptions, EmbedRequest, EmbedResponse


class TestHealthEndpoint:
    """Test cases for the health endpoint."""
    
    def test_health_endpoint_success(self):
        """Test successful health check response."""
        with patch('app.model_info', '/models/3b'), \
             patch('app.DEVICE', 'cpu'), \
             patch('app.MODEL_DIR', '/models/3b'):
            
            client = TestClient(app)
            response = client.get("/healthz")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["model"] == "/models/3b"
            assert data["device"] == "cpu"
            assert data["offline"] is True


class TestTextEmbeddings:
    """Test cases for text embedding functionality."""
    
    @pytest.fixture
    def mock_model_and_processor(self):
        """Mock the model and processor for testing."""
        with patch('app.model') as mock_model, \
             patch('app.processor') as mock_processor:
            
            # Mock model output
            mock_output = torch.randn(1, 4, 128)  # batch_size=1, num_vectors=4, dim=128
            mock_model.return_value = mock_output
            mock_model.device = torch.device('cpu')
            
            # Mock processor
            mock_processor.process_queries.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            yield mock_model, mock_processor
    
    def test_text_embedding_col_variant(self, mock_model_and_processor):
        """Test text embedding with col variant."""
        mock_model, mock_processor = mock_model_and_processor
        
        client = TestClient(app)
        payload = {
            "input": {
                "texts": ["Hello world", "Test text"]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "usage" in data
        assert data["variant"] == "col"
        assert data["model"] == "/models/3b"
        assert len(data["data"]) == 2  # Two text inputs
    
    def test_text_embedding_dense_variant(self, mock_model_and_processor):
        """Test text embedding with dense variant."""
        mock_model, mock_processor = mock_model_and_processor
        
        client = TestClient(app)
        payload = {
            "input": {
                "texts": ["Hello world"]
            },
            "options": {
                "variant": "dense",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["variant"] == "dense"
        assert len(data["data"]) == 1
        assert len(data["data"][0]) == 128  # Dense vector should be 128-dimensional
    
    def test_text_embedding_empty_batch(self):
        """Test error handling for empty text batch."""
        client = TestClient(app)
        payload = {
            "input": {
                "texts": []
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 413
        assert "Batch 1..64" in response.json()["detail"]
    
    def test_text_embedding_too_long(self):
        """Test error handling for text that's too long."""
        client = TestClient(app)
        long_text = "a" * 3000  # Exceeds MAX_TEXT_LEN
        payload = {
            "input": {
                "texts": [long_text]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 413
        assert "text too long" in response.json()["detail"]
    
    def test_text_embedding_batch_too_large(self):
        """Test error handling for batch that's too large."""
        client = TestClient(app)
        payload = {
            "input": {
                "texts": [f"text {i}" for i in range(100)]  # Exceeds MAX_BATCH
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 413
        assert "Batch 1..64" in response.json()["detail"]


class TestImageEmbeddings:
    """Test cases for image embedding functionality."""
    
    @pytest.fixture
    def mock_model_and_processor(self):
        """Mock the model and processor for testing."""
        with patch('app.model') as mock_model, \
             patch('app.processor') as mock_processor:
            
            # Mock model output
            mock_output = torch.randn(1, 4, 128)  # batch_size=1, num_vectors=4, dim=128
            mock_model.return_value = mock_output
            mock_model.device = torch.device('cpu')
            
            # Mock processor
            mock_processor.process_images.return_value = {
                'pixel_values': torch.randn(1, 3, 224, 224)
            }
            
            yield mock_model, mock_processor
    
    def create_test_image_b64(self, width=64, height=64, color='red') -> str:
        """Create a test image and return as base64 string."""
        img = Image.new('RGB', (width, height), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def test_image_embedding_success(self, mock_model_and_processor):
        """Test successful image embedding."""
        mock_model, mock_processor = mock_model_and_processor
        
        client = TestClient(app)
        test_image = self.create_test_image_b64()
        
        payload = {
            "input": {
                "image_b64": [test_image]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["variant"] == "col"
        assert len(data["data"]) == 1
    
    def test_image_embedding_multiple_images(self, mock_model_and_processor):
        """Test embedding multiple images."""
        mock_model, mock_processor = mock_model_and_processor
        
        client = TestClient(app)
        test_images = [
            self.create_test_image_b64(64, 64, 'red'),
            self.create_test_image_b64(64, 64, 'blue')
        ]
        
        payload = {
            "input": {
                "image_b64": test_images
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
    
    def test_image_embedding_empty_batch(self):
        """Test error handling for empty image batch."""
        client = TestClient(app)
        payload = {
            "input": {
                "image_b64": []
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 413
        assert "Batch 1..64" in response.json()["detail"]
    
    def test_image_embedding_invalid_base64(self):
        """Test error handling for invalid base64 data."""
        client = TestClient(app)
        payload = {
            "input": {
                "image_b64": ["invalid_base64_data"]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 400
        assert "Invalid image data" in response.json()["detail"]
    
    def test_image_embedding_urls_disabled(self):
        """Test that image URLs are properly disabled."""
        client = TestClient(app)
        payload = {
            "input": {
                "image_urls": ["https://example.com/image.jpg"]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 400
        assert "image_urls disabled; send base64" in response.json()["detail"]
    
    def test_image_embedding_small_image_resize(self, mock_model_and_processor):
        """Test that small images are properly resized."""
        mock_model, mock_processor = mock_model_and_processor
        
        client = TestClient(app)
        # Create a 1x1 pixel image (should be resized to 32x32)
        small_image = self.create_test_image_b64(1, 1, 'red')
        
        payload = {
            "input": {
                "image_b64": [small_image]
            },
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


class TestAuthentication:
    """Test cases for authentication functionality."""
    
    def test_no_auth_key_required_when_not_set(self):
        """Test that no auth key is required when INTERNAL_KEY is not set."""
        with patch('app.INTERNAL_KEY', None):
            client = TestClient(app)
            payload = {
                "input": {"texts": ["test"]},
                "options": {"variant": "col", "normalize": True}
            }
            
            response = client.post("/v1/embed", json=payload)
            # Should not fail due to auth (might fail for other reasons like missing model)
            assert response.status_code != 401
    
    def test_auth_key_required_when_set(self):
        """Test that auth key is required when INTERNAL_KEY is set."""
        with patch('app.INTERNAL_KEY', 'test-key'):
            client = TestClient(app)
            payload = {
                "input": {"texts": ["test"]},
                "options": {"variant": "col", "normalize": True}
            }
            
            response = client.post("/v1/embed", json=payload)
            assert response.status_code == 401
            assert "unauthorized" in response.json()["detail"]
    
    def test_valid_auth_key(self):
        """Test that valid auth key is accepted."""
        with patch('app.INTERNAL_KEY', 'test-key'):
            client = TestClient(app)
            payload = {
                "input": {"texts": ["test"]},
                "options": {"variant": "col", "normalize": True}
            }
            
            headers = {"x-internal-key": "test-key"}
            response = client.post("/v1/embed", json=payload, headers=headers)
            # Should not fail due to auth (might fail for other reasons like missing model)
            assert response.status_code != 401


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_normalize_function(self):
        """Test the _normalize utility function."""
        from app import _normalize
        
        # Test with a simple tensor
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        normalized = _normalize(tensor)
        
        # Check that the result is normalized (L2 norm = 1)
        norm = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)
    
    def test_pool_to_dense_function(self):
        """Test the _pool_to_dense utility function."""
        from app import _pool_to_dense
        
        # Test with 3D tensor (batch_size, num_vectors, dim)
        tensor_3d = torch.randn(1, 4, 128)
        pooled = _pool_to_dense(tensor_3d)
        
        assert pooled.shape == (128,)  # Should be squeezed to 1D
        
        # Test with 2D tensor (num_vectors, dim)
        tensor_2d = torch.randn(4, 128)
        pooled_2d = _pool_to_dense(tensor_2d)
        
        assert pooled_2d.shape == (1, 128)  # Should keep batch dimension
    
    def test_load_b64_function(self):
        """Test the _load_b64 utility function."""
        from app import _load_b64
        
        # Create a test image
        img = Image.new('RGB', (64, 64), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Test loading
        loaded_images = _load_b64([img_b64])
        
        assert len(loaded_images) == 1
        assert isinstance(loaded_images[0], Image.Image)
        assert loaded_images[0].size == (64, 64)
        assert loaded_images[0].mode == 'RGB'
    
    def test_load_b64_small_image_resize(self):
        """Test that _load_b64 resizes small images."""
        from app import _load_b64
        
        # Create a small test image
        img = Image.new('RGB', (1, 1), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Test loading
        loaded_images = _load_b64([img_b64])
        
        assert len(loaded_images) == 1
        assert loaded_images[0].size == (32, 32)  # Should be resized to minimum size


class TestModelLoading:
    """Test cases for model loading functionality."""
    
    def test_model_path_resolution(self):
        """Test that model path is resolved correctly."""
        with patch('app.MODEL_DIR', '/models/3b'), \
             patch('app.os.path.exists', return_value=True):
            
            from app import model_path
            assert model_path == '/models/3b'
    
    def test_model_path_fallback(self):
        """Test that model path falls back to MODEL_ID when MODEL_DIR doesn't exist."""
        with patch('app.MODEL_DIR', None), \
             patch('app.MODEL_ID', 'nomic-ai/test-model'):
            
            from app import model_path
            assert model_path == 'nomic-ai/test-model'
    
    def test_lora_model_detection(self):
        """Test LoRA model detection logic."""
        with patch('app.MODEL_DIR', '/models/3b'), \
             patch('app.os.path.exists') as mock_exists, \
             patch('app.json.load') as mock_json_load:
            
            # Mock adapter config
            mock_adapter_config = {
                "base_model_name_or_path": "vidore/colqwen2.5-base"
            }
            mock_json_load.return_value = mock_adapter_config
            
            # Mock file existence
            def exists_side_effect(path):
                if path == '/models/3b/adapter_config.json':
                    return True
                elif path == '/models/base':
                    return True
                return False
            
            mock_exists.side_effect = exists_side_effect
            
            # Test the detection logic
            from app import is_lora_model, base_model_path
            assert is_lora_model is True
            assert base_model_path == '/models/base'


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_invalid_json_payload(self):
        """Test handling of invalid JSON payload."""
        client = TestClient(app)
        
        response = client.post("/v1/embed", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        client = TestClient(app)
        
        # Missing 'input' field
        payload = {
            "options": {
                "variant": "col",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        assert response.status_code == 422
    
    def test_invalid_variant(self):
        """Test handling of invalid variant."""
        client = TestClient(app)
        
        payload = {
            "input": {
                "texts": ["test"]
            },
            "options": {
                "variant": "invalid_variant",
                "normalize": True
            }
        }
        
        response = client.post("/v1/embed", json=payload)
        # Should still work as variant is just a string field
        assert response.status_code != 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
