import torch
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from torchtmpl.data import get_dataloaders
from torchtmpl.models import get_model

def generate_submission(config_path, model_path=None):
    """Generate submission.csv from a trained model"""
    # Load config
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    # Get model path
    if model_path is None:
        model_path = config['logging']['model_path']
    
    print(f"Loading model from: {model_path}")
    
    # Get test dataloader
    _, _, test_loader = get_dataloaders(config)
    
    # Load model
    model = get_model(config)
    model.load_state_dict(torch.load(model_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    predictions = []
    image_ids = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, ids = batch
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy().tolist())
            image_ids.extend(ids)
    
    # Create submission
    submission = pd.DataFrame({
        'id': image_ids,
        'class': predictions
    })
    
    # Save submission
    output_path = Path(config['logging']['results_path']) / 'submission.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m torchtmpl.inference config.yaml [model_path]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_submission(config_path, model_path)