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
    
    cfg = {"class": "CNN"}
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

def test_dinov3():
    """Test DINOv3 model with dummy config"""
    print("\n" + "="*60)
    print("Testing DINOv3Classifier...")
    
    cfg = {
        "class": "DINOv3Classifier",
        "model_size": "small",  # Options: 'small', 'base', 'large', 'giant'
        "freeze_backbone": True,
        "head_hidden_dims": [512, 256],
        "dropout": 0.3
    }
    
    input_size = (3, 224, 224)  # DINOv3 expects 224x224
    batch_size = 4  # Smaller batch for testing
    num_classes = 18
    
    print(f"Config: {cfg}")
    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    try:
        print("\nBuilding model...")
        model = build_model(cfg, input_size, num_classes)
        
        print("\nModel built successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        input_tensor = torch.randn(batch_size, *input_size)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output tensor shape: {output.shape}")
        print(f"   Expected: [{batch_size}, {num_classes}]")
        
        assert output.shape == (batch_size, num_classes), \
            f"Shape mismatch! Got {output.shape}, expected ({batch_size}, {num_classes})"
        
        print("\nAll tests passed!")
        
        # Additional checks
        print("\nModel Statistics:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Frozen params: {total_params - trainable_params:,}")
        
    except Exception as e:
        print(f"\nTest failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise



if __name__ == "__main__":
    test_linear()
    test_cnn()
    test_custom_cnn()
    test_dinov3()
