# Pixel Revival Backend API

Flask-based backend API for EDSR image denoising and super-resolution.

## Features

- EDSR-baseline model for 4x super-resolution
- CPU-optimized inference
- RESTful API with image upload
- CORS enabled for frontend integration

## Project Structure

```
backend/
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── inference.py       # EDSR inference wrapper
│   ├── model/
│   │   ├── __init__.py
│   │   ├── common.py      # Basic network blocks
│   │   └── edsr.py        # EDSR architecture
│   └── data/
│       ├── __init__.py
│       └── common.py      # Image preprocessing
├── models/
│   └── edsr_baseline_x4-6b446fab.pt  # Pretrained weights (5.8MB)
└── storage/
    ├── uploads/           # Temporary uploaded images
    └── outputs/           # Processed results
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
python app.py
```

Server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```
GET /
```

Response:
```json
{
  "status": "ok",
  "message": "Pixel Revival API is running",
  "version": "1.0.0"
}
```

#### 2. Denoise Image
```
POST /api/denoise
Content-Type: multipart/form-data
Body: image file
```

Response:
```json
{
  "success": true,
  "message": "Image processed successfully",
  "input": {
    "filename": "example.jpg",
    "size": [640, 480],
    "url": "/api/images/xxx_input.jpg"
  },
  "output": {
    "filename": "xxx_output.png",
    "size": [2560, 1920],
    "url": "/api/images/xxx_output.png"
  },
  "processing_time": "2.34s"
}
```

#### 3. Get Image
```
GET /api/images/<filename>
```

Returns the image file.

#### 4. Cleanup (Optional)
```
POST /api/cleanup
```

Removes files older than 1 hour from storage.

## Model Details

- **Model**: EDSR-baseline
- **Parameters**:
  - ResBlocks: 16
  - Feature channels: 64
  - Scale: 4x
- **Input**: RGB images (any size)
- **Output**: 4x upscaled images

## Performance

- **CPU**: ~2-5 seconds per image (depending on size)
- **Memory**: ~500MB RAM
- **Model size**: 5.8MB

## Deployment

### Local Development
```bash
python app.py
```

### Production (AWS/Cloud)

1. Use a WSGI server (e.g., Gunicorn):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. For better performance, consider:
   - AWS EC2 t3.medium or larger
   - CPU optimization with `torch.set_num_threads()`
   - Add reverse proxy (Nginx)
   - Enable file cleanup cron job

### Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- TIFF

Max file size: 16MB

## Notes

- First request will take longer (model loading)
- Subsequent requests are faster (model cached in memory)
- For GPU acceleration, change `device='cpu'` to `device='cuda'` in app.py
- Files in storage/ folders are temporary and can be cleaned periodically
