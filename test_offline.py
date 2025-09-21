#!/usr/bin/env python3
"""
Test script to verify offline model loading works correctly.
This script simulates the same environment as the Docker container.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Set up offline environment
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

def test_model_loading():
    """Test that the model can be loaded in offline mode."""
    print("Testing offline model loading...")
    
    # Simulate the Docker environment
    model_dir = "/models/3b"  # This would be set in Docker
    fallback_model_id = "nomic-ai/colnomic-embed-multimodal-3b"
    
    # Determine which path to use
    if os.path.exists(model_dir):
        model_path = model_dir
        print(f"✓ Using local model directory: {model_path}")
    else:
        model_path = fallback_model_id
        print(f"⚠ Using fallback model ID: {model_path}")
        print("  (This will fail in offline mode - expected behavior)")
    
    try:
        # Test the same imports as app.py
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        import torch
        
        print(f"✓ Successfully imported ColPali modules")
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test model loading (this will fail if not offline-ready)
        print(f"Attempting to load model from: {model_path}")
        
        model = ColQwen2_5.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU for testing
        )
        print("✓ Model loaded successfully in offline mode!")
        
        processor = ColQwen2_5_Processor.from_pretrained(
            model_path, 
            local_files_only=True
        )
        print("✓ Processor loaded successfully in offline mode!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        if "We couldn't connect to" in str(e):
            print("  This is expected if MODEL_DIR is not set or model not downloaded.")
            print("  In Docker, the model should be pre-downloaded to /models/3b")
        return False

def test_environment_variables():
    """Test that environment variables are set correctly."""
    print("\nTesting environment variables...")
    
    required_vars = [
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY"
    ]
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}={value}")
        else:
            print(f"✗ {var} not set")

def main():
    print("=== Offline Model Loading Test ===")
    test_environment_variables()
    success = test_model_loading()
    
    if success:
        print("\n🎉 All tests passed! The app should work in offline mode.")
    else:
        print("\n❌ Tests failed. Check the Docker build process.")
        sys.exit(1)

if __name__ == "__main__":
    main()
