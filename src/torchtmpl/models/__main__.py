# coding: utf-8

# External imports
import torch
import timm

# Local imports
from . import build_model



def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

def test_custom_cnn():
    """Test the custom CNN model"""
    print("\n" + "="*60)
    print("Testing Custom CNN...")
    
    cfg = {"class": "CNNmultilayer"}
    cfg = {**cfg, "num_layers": 3, "num_kernels": 32}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    
    print(f"Config: {cfg}")
    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    try:
        print("\nBuilding model...")
        model = build_model(cfg, input_size, num_classes)
        
        print("Model built successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        input_tensor = torch.randn(batch_size, *input_size)
        output = model(input_tensor)
        
        print(f"Output tensor shape: {output.shape}")
        print(f"   Expected: [{batch_size}, {num_classes}]")
        
        assert output.shape == (batch_size, num_classes), \
            f"Shape mismatch! Got {output.shape}, expected ({batch_size}, {num_classes})"
        
        print("\nTest passed!")
        
    except Exception as e:
        print(f"\nTest failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise


def test_resnet50():
    """Test the ResNet50 model"""
    print("\n" + "="*60)
    print("Testing ResNet50...")
    
    cfg = {"class": "ResNet50"}
    input_size = (3, 64, 64)
    batch_size = 32
    num_classes = 10
    
    print(f"Config: {cfg}")
    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    try:
        print("\nBuilding model...")
        model = build_model(cfg, input_size, num_classes)
        
        print("Model built successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        input_tensor = torch.randn(batch_size, *input_size)
        output = model(input_tensor)
        
        print(f"Output tensor shape: {output.shape}")
        print(f"   Expected: [{batch_size}, {num_classes}]")
        
        assert output.shape == (batch_size, num_classes), \
            f"Shape mismatch! Got {output.shape}, expected ({batch_size}, {num_classes})"
        
        print("\nTest passed!")
        
    except Exception as e:
        print(f"\nTest failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_linear()
    test_cnn()
    test_custom_cnn()
    test_resnet50()
    print("\n" + "="*60 + "\nAll tests completed.")
