# LearningASL

A deep learning project for American Sign Language (ASL) recognition, featuring both static letter classification and dynamic word recognition from video sequences.

## Overview

This project implements two complementary approaches to ASL recognition:

1. **Letter Classification (CNN)**: A ResNet50-based CNN that classifies static hand gesture images into 28 classes (A-Z, DEL, SPACE)

2. **Word Recognition (CNN + Transformer)**: An end-to-end model that processes video sequences to recognize ASL words using CNN feature extraction combined with a Transformer for temporal modeling

## Project Structure

```
LearningASL/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_cnn_letter_classification.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_demo_cnn_transformer.ipynb
│   └── 04_train_workflow.ipynb
│
├── src/                          # Source code
│   ├── __init__.py               # Package exports
│   ├── cnn_model.py              # CNN model definitions
│   ├── model.py                  # Transformer model definitions
│   ├── dataset.py                # Dataset classes
│   ├── extract_keypoints.py      # MediaPipe keypoint extraction
│   ├── prepare_frames.py         # Video frame extraction pipeline
│   └── train_ptn.py              # Training script for Pose Transformer
│
├── models/                       # Saved model weights
│   └── best_cnn_params.pth
│
├── images_folder/                # ASL letter images (A-Z, DEL, SPACE)
├── videos_folder/                # Source videos (git-ignored)
├── words/                        # Extracted video frames (git-ignored)
├── keypoints_data/               # Extracted keypoints (git-ignored)
│
└── unigram_freq.csv              # Word frequency data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/madelyn-redick/LearningASL.git
cd LearningASL

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Run the data preprocessing pipeline to download videos and extract frames:

```bash
python src/prepare_frames.py
```

Extract hand keypoints from frames using MediaPipe:

```bash
python src/extract_keypoints.py
```

### 2. Training

Train the Pose Transformer Network on keypoint sequences:

```bash
python src/train_ptn.py --epochs 50 --batch_size 32 --lr 1e-4
```

### 3. Using the Models

```python
from src import ASL_CNN, ASL_CNN_FeatureExtractor, PoseTransformer, ASLSequenceModel

# Load pre-trained CNN for letter classification
cnn = ASL_CNN(num_classes=28)
cnn.load_state_dict(torch.load('models/best_cnn_params.pth'))

# Create feature extractor from trained CNN
feature_extractor = ASL_CNN_FeatureExtractor(cnn)

# Create end-to-end word recognition model
transformer = PoseTransformer(input_dim=512, num_classes=40)
model = ASLSequenceModel(feature_extractor, transformer)
```

## Model Architecture

### CNN (Letter Classification)
- **Base**: ResNet50 (pretrained on ImageNet)
- **Fine-tuning**: Last residual block unfrozen
- **Head**: Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→28)
- **Input**: RGB images (128×128)
- **Output**: 28 classes (A-Z, DEL, SPACE)

### Pose Transformer (Word Recognition)
- **Input Projection**: Linear(input_dim → d_model)
- **Positional Encoding**: Sinusoidal
- **Encoder**: N Transformer encoder layers
- **Pooling**: Global average over time dimension
- **Classifier**: Linear(d_model→256) → ReLU → Dropout → Linear(256→num_classes)

### End-to-End Model
1. **Frame Encoding**: Each video frame → CNN → 512-dim embedding
2. **Temporal Modeling**: Sequence of embeddings → Transformer → logits
3. **Prediction**: Softmax over word classes

## Data Sources

- **Letter Images**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Word Videos**: [WLASL Dataset](https://www.kaggle.com/datasets/waseemnagahhenes/sign-language-dataset-wlasl-videos)
- **Word Frequencies**: [English Word Frequency](https://www.kaggle.com/datasets/rtatman/english-word-frequency)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_cnn_letter_classification.ipynb` | Train and evaluate CNN for letter recognition |
| `02_data_preprocessing.ipynb` | Interactive data exploration and preprocessing |
| `03_demo_cnn_transformer.ipynb` | Demonstrate the end-to-end CNN+Transformer model |
| `04_train_workflow.ipynb` | Complete training workflow (Colab-ready) |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MediaPipe
- OpenCV
- See `requirements.txt` for full list

## License

This project is for educational purposes.

## Acknowledgments

- [WLASL Dataset](https://github.com/dxli94/WLASL) for word-level ASL videos
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) for letter images
