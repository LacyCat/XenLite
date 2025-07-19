# XenLite Xen3 - Tiny Diffusion Model

A high-performance, lightweight diffusion model optimized for CIFAR-10 image generation. Designed for efficient inference on modest hardware (i7 1065g7) with focus on speed over quality.

## Features

- **Tiny Architecture**: Optimized UNet with only ~3M parameters
- **Fast Inference**: DDIM sampling with 50 steps for quick generation
- **Class-Conditional**: Generate specific CIFAR-10 classes
- **Web API**: Beautiful HTML interface for easy image generation
- **Performance Focused**: Prioritizes speed and efficiency over image quality

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

**Windows:**
```cmd
# Quick training (recommended for testing)
python train.py --num_epochs 20 --batch_size 32

# Full training
run_training.bat
```

**Linux/Mac:**
```bash
# Quick training (recommended for testing)
python train.py --num_epochs 20 --batch_size 32

# Full training
./run_training.sh
```

### 3. Inference API

**Windows:**
```cmd
# Start the web API
run_api.bat

# Or manually
python api.py --model_path ./models/xenlite_xen3_final.pth
```

**Linux/Mac:**
```bash
# Start the web API
./run_api.sh

# Or manually
python api.py --model_path ./models/xenlite_xen3_final.pth
```

Visit `http://127.0.0.1:5000` for the web interface.

## CIFAR-10 Classes

- airplane ✈️
- automobile 🚗  
- bird 🐦
- cat 🐱
- deer 🦌
- dog 🐕
- frog 🐸
- horse 🐎
- ship 🚢
- truck 🚚

## Model Architecture

The XenLite Xen3 uses a streamlined UNet architecture:

- **Encoder**: 3 downsampling blocks with residual connections
- **Middle**: Attention and residual blocks
- **Decoder**: 3 upsampling blocks with skip connections  
- **Timestep Embedding**: Sinusoidal position embeddings
- **Class Conditioning**: Learnable class embeddings

## Performance Targets

- **Training Time**: 1-4 hours on GTX 1660 Ti
- **Inference Time**: ~2-5 seconds per image on i7 1065g7
- **Image Size**: 32x32 (upscaled to 256x256 for display)
- **Model Size**: ~3M parameters for fast loading

## Usage Examples

### Training Custom

```python
from XenLiteXen3 import XenLiteXen3, DDPMScheduler

model = XenLiteXen3(model_channels=64, num_classes=10)
scheduler = DDPMScheduler(num_timesteps=1000)
```

### API Generation

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"class_name": "cat", "num_inference_steps": 50}'
```

## File Structure

```
xen3/
├── XenLiteXen3.py      # Model architecture
├── train.py            # Training script  
├── api.py              # Web API server
├── requirements.txt    # Dependencies
├── run_training.sh     # Training launcher (Linux/Mac)
├── run_training.bat    # Training launcher (Windows)
├── run_api.sh          # API launcher (Linux/Mac)
├── run_api.bat         # API launcher (Windows)
└── models/             # Saved models
```

## Hardware Requirements

**Training:**
- GPU: GTX 1660 Ti or equivalent
- RAM: 8GB minimum
- Storage: 2GB for dataset + models

**Inference:**
- CPU: i7 1065g7 or equivalent  
- RAM: 4GB minimum
- No GPU required for inference

## Generation Quality vs Speed

The model prioritizes speed over quality. For higher quality:
- Increase `num_inference_steps` (50-100)
- Use larger `model_channels` (128 instead of 64)
- Train for more epochs

For faster inference:
- Reduce `num_inference_steps` (10-25)
- Use smaller model channels
- Implement fp16 inference

---

**XenLite Xen3** - Third generation of the XenLite series, optimized for performance and efficiency.