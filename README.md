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
├── frontend/               # Next.js (port 3000)
│   └── app/
│       ├── page.tsx        # Home page
│       └── demo/           # Demo page
└── backend/                # Flask API (port 5000)
    ├── app.py              # API server
    ├── src/
    │   ├── inference.py               # EDSR inference
    │   ├── realesrgan_inference.py    # Real-ESRGAN inference
    │   ├── preprocessing.py           # Image preprocessing
    │   ├── degradation.py             # Image degradation for evaluation
    │   └── metrics.py                 # Evaluation metrics
    ├── models/             # Model weights
    ├── storage/            # Upload/output storage
    └── test_*.py           # Testing scripts
```

## API Endpoints

- `POST /api/pipeline` - Full enhancement pipeline
- `POST /api/denoise` - EDSR only
- `GET /api/images/<filename>` - Retrieve processed image
- `POST /api/cleanup` - Clear storage

## Evaluation Metrics

- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **NIQE** - Natural Image Quality Evaluator (no-reference)
- **LPIPS** - Learned Perceptual Image Patch Similarity

## Requirements

- Python 3.8+
- PyTorch 2.4+ with CUDA
- Node.js 18+

## Models

| Model       | Scale | Purpose                     |
| ----------- | ----- | --------------------------- |
| Real-ESRGAN | 4x    | Main super-resolution       |
| GFPGAN      | -     | Face enhancement (optional) |
| EDSR        | 4x    | Post-processing             |

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
