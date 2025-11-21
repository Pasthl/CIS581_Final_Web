# Pixel Revival

CIS5810 - Fall25 Final Project

**Authors**: Ann Hua, Livia Yuan

Image enhancement web application with preprocessing, Real-ESRGAN super-resolution, and EDSR post-processing.

## Pipeline

1. **Preprocessing** - Contrast enhancement (CLAHE)
2. **Real-ESRGAN (4x)** - Super-resolution with optional GFPGAN face enhancement
3. **EDSR (4x)** - Post-processing super-resolution

## Quick Start

### Backend

```bash
cd backend

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download models to backend/models/
# - edsr_baseline_x4-6b446fab.pt
# - RealESRGAN_x4plus.pth

python app.py
```

Backend runs at http://localhost:5000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

## Project Structure

```
CIS581_Final_Web/
├── frontend/           # Next.js (port 3000)
│   └── app/demo/       # Main demo page
└── backend/            # Flask API (port 5000)
    ├── app.py          # API server
    ├── src/
    │   ├── inference.py           # EDSR wrapper
    │   ├── realesrgan_inference.py # Real-ESRGAN wrapper
    │   └── preprocessing.py       # Image preprocessing
    └── models/         # Model weights
```

## API

- `POST /api/pipeline` - Process image through pipeline
- `POST /api/denoise` - EDSR only

## Requirements

- Python 3.8+
- PyTorch 2.4+ with CUDA
- Node.js 18+

## Models

| Model | Scale | Purpose |
|-------|-------|---------|
| Real-ESRGAN | 4x | Main super-resolution |
| GFPGAN | - | Face enhancement (optional) |
| EDSR | 4x | Post-processing |

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
