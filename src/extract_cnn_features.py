import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cnn_model import ASL_CNN, ASL_CNN_FeatureExtractor

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_features(model, device, video_dir, transform):
    """
    Extracts features for all frames in a video directory.
    Returns: numpy array of shape (num_frames, feature_dim)
    """
    # Get all frame images
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not frame_paths:
        return None

    features_list = []
    
    # Process in batches for efficiency (optional, but good for speed)
    # For simplicity, we'll do one by one or small batches. 
    # Let's do batch size of 1 for simplicity of implementation unless slow.
    # Actually, batching is much faster. Let's do batch size 32.
    
    batch_size = 32
    current_batch = []
    
    with torch.no_grad():
        for i, frame_path in enumerate(frame_paths):
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = transform(img)
                current_batch.append(img_tensor)
                
                if len(current_batch) == batch_size or i == len(frame_paths) - 1:
                    batch_tensor = torch.stack(current_batch).to(device)
                    batch_features = model(batch_tensor)
                    features_list.append(batch_features.cpu().numpy())
                    current_batch = []
            except Exception as e:
                print(f"Error reading {frame_path}: {e}")
                continue

    if not features_list:
        return None

    return np.concatenate(features_list, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Extract CNN features from video frames")
    parser.add_argument("--data_root", type=str, default="words", help="Path to words directory containing frames")
    parser.add_argument("--output_root", type=str, default="cnn_features", help="Path to save features")
    parser.add_argument("--model_path", type=str, default="models/best_cnn_params.pth", help="Path to trained CNN model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    # We need to instantiate the model with the correct number of classes used during training.
    # Usually this is 28 or 29 depending on dataset. The checkpoint should match.
    # However, for feature extraction, we drop the last layer anyway.
    # We just need to load the weights successfully.
    # Let's try to infer num_classes or default to 28/29.
    # The saved state_dict will complain if sizes don't match.
    # Inspecting src/train_cnn.py, it uses ImageFolder on images_folder which likely has 29 classes (A-Z, del, space, nothing).
    # Let's try to load it. If it fails, we might need to check the checkpoint.
    # A safe bet is to load the state dict and check the size of the final layer if needed, 
    # but let's assume standard training for now.
    
    print(f"Loading model from {args.model_path}...")
    try:
        # First try initializing with common class counts if load fails
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Check output size of fc.3.weight or similar to guess classes if needed
        # Structure: _base_model.fc.3.weight
        if '_base_model.fc.3.weight' in checkpoint:
            out_features = checkpoint['_base_model.fc.3.weight'].shape[0]
            print(f"Detected {out_features} classes from checkpoint.")
            num_classes = out_features
        else:
            num_classes = 29 # Fallback
            
        base_cnn = ASL_CNN(num_classes=num_classes)
        base_cnn.load_state_dict(checkpoint)
        base_cnn.to(device)
        base_cnn.eval()
        
        feature_extractor = ASL_CNN_FeatureExtractor(base_cnn)
        feature_extractor.to(device)
        feature_extractor.eval()
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run src/train_cnn.py first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Transforms (Must match training transforms)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Iterate and Extract
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            continue
            
        print(f"Processing split: {split}")
        
        # Iterate over words
        words = sorted(os.listdir(split_dir))
        for word in words:
            word_dir = os.path.join(split_dir, word)
            if not os.path.isdir(word_dir):
                continue
                
            # Iterate over videos
            videos = sorted(os.listdir(word_dir))
            for video_id in videos:
                video_dir = os.path.join(word_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue
                    
                # Setup output path
                output_dir = os.path.join(args.output_root, split, word)
                ensure_dir(output_dir)
                output_path = os.path.join(output_dir, f"{video_id}.npy")
                
                if os.path.exists(output_path):
                    continue
                    
                # Extract
                features = extract_features(feature_extractor, device, video_dir, transform)
                
                if features is not None:
                    np.save(output_path, features)
                    # print(f"Saved {output_path} shape={features.shape}")
                else:
                    print(f"Warning: No features extracted for {video_dir}")

    print("Feature extraction complete.")

if __name__ == "__main__":
    main()
