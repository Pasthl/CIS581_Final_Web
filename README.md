# Pixel Revival

CIS5810 - Fall25 Final Project

**Authors**: Ann Hua, Livia Yuan

A full-stack web application for image processing. The project focuses on **image deblurring and enhancement**, with deblurring algorithm as main process, EDSR-based super-resolution as a post-processing step.

## Project Overview

This application is designed for:

1. **Primary Goal**: Image deblurring (currently in research/development)
2. **Post-processing**: EDSR super-resolution for enhanced image quality
3. **Pipeline**: Blur → Deblur (main processing) → Super-resolution (EDSR)

The EDSR module implemented here serves as the final enhancement stage after the main deblurring algorithm.

## Architecture

### Frontend-Backend Communication

```
┌─────────────────────────────────────────────┐
│  Browser (localhost:3000)                   │
│  ┌───────────────────────────────────┐     │
│  │  Next.js Frontend                  │     │
│  │  - UI/UX                           │     │
│  │  - User interaction                │     │
│  └──────────────┬─────────────────────┘     │
└─────────────────┼───────────────────────────┘
                  │ HTTP fetch()
                  ▼
         ┌────────────────────┐
         │  localhost:5000    │ ← Flask API Server
         └────────┬───────────┘
┌─────────────────┼───────────────────────────┐
│                 ▼                            │
│  ┌────────────────────────────────┐         │
│  │  Flask Backend API              │         │
│  │  - Image upload endpoint        │         │
│  │  - EDSR inference               │         │
│  │  - Return processed results     │         │
│  └────────┬────────────────────────┘         │
│           ▼                                  │
│  ┌────────────────┐                         │
│  │  EDSR Model    │                         │
│  │  (PyTorch)     │                         │
│  └────────────────┘                         │
└──────────────────────────────────────────────┘
```

**Key Points**:

- Frontend (port 3000): User interface, runs in browser
- Backend (port 5000): API server, runs on your machine/server
- Communication: Frontend makes HTTP requests to backend
- Why two servers? Separation of concerns - UI vs. processing logic

## Features

- 🎨 Modern, responsive web interface
- 🖼️ Image upload and real-time preview
- ✨ EDSR 4x super-resolution (post-processing)
- 🌐 Full-stack: Next.js frontend + Flask backend
- 📊 Processing status and timing display

## Project Structure

```
CIS581_Final_Web/
├── frontend/                    # Next.js web application (port 3000)
│   ├── app/
│   │   ├── page.tsx            # Home page
│   │   ├── demo/page.tsx       # Demo page with image upload
│   │   └── globals.css         # Styles
│   └── package.json
│
└── backend/                     # Flask API + EDSR model (port 5000)
    ├── app.py                  # API server entry point
    ├── src/
    │   ├── inference.py        # EDSR inference wrapper
    │   ├── model/              # EDSR architecture
    │   │   ├── common.py       # ResBlock, Upsampler
    │   │   └── edsr.py         # EDSR model
    │   └── data/               # Image preprocessing
    │       └── common.py       # np2Tensor utilities
    ├── models/                 # Pretrained weights (5.8MB)
    │   └── edsr_baseline_x4-6b446fab.pt
    ├── storage/                # Temporary storage
    │   ├── uploads/            # Input images
    │   └── outputs/            # Processed results
    └── requirements.txt        # Python dependencies
```

## Setup

### Backend Setup

1. Create **virtual** environment (recommended):

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Test the setup:

```bash
python quick_test.py
```

4. Start the API server:

```bash
python app.py
```

Backend API will be available at [http://localhost:5000](http://localhost:5000)

### Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Run development server:

```bash
npm run dev
```

Frontend will be available at [http://localhost:3000](http://localhost:3000)

## Usage

1. **Start both servers** (backend on 5000, frontend on 3000)
2. Navigate to [http://localhost:3000](http://localhost:3000)
3. Click "Experience the Magic" to enter demo page
4. Upload an image
5. View original and EDSR-enhanced (4x) results side-by-side

## API Endpoints

Backend provides RESTful API:

- `GET /` - Health check
- `POST /api/denoise` - Upload image and get EDSR result
- `GET /api/images/<filename>` - Retrieve processed images

See [backend/README.md](backend/README.md) for detailed API documentation.

## Current Implementation Status

### ✅ Completed (EDSR Post-processing)

- EDSR model integration
- 4x super-resolution
- Frontend-backend communication
- Image upload/download pipeline

### 🚧 In Progress (Main Processing)

- Image deblurring algorithm (research phase)
- Main processing pipeline integration

## EDSR Model Details

- **Architecture**: EDSR-baseline
- **Purpose**: Post-processing super-resolution
- **Scale**: 4x upscaling
- **Parameters**: 16 ResBlocks, 64 features
- **Model Size**: 5.8MB
- **Performance**: ~2-5s per image on CPU

## Technologies

### Frontend

- Next.js 14
- React 18
- TypeScript
- CSS (custom styling)

### Backend

- Flask (API server)
- PyTorch 2.x (EDSR inference)
- Pillow (image processing)
- NumPy

## Git Configuration

The project uses multiple `.gitignore` files:

- Root `.gitignore` - Overall project exclusions
- `backend/.gitignore` - Python/Flask specific
- Frontend uses Next.js default ignores

## Deployment

### Local Development

Follow setup instructions above.

### Production Considerations

- Frontend: Deploy to Vercel/Netlify
- Backend: Deploy to AWS EC2, Heroku, or similar
- For AWS: t3.medium or larger recommended
- GPU optional (CPU optimized for now)

See [backend/README.md](backend/README.md) for deployment details.

## Future Work

- [ ] Integrate main deblurring algorithm
- [ ] Pipeline: Blur → Deblur → EDSR enhancement
- [ ] Performance optimization
- [ ] Batch processing support
- [ ] Model comparison features

## License

Educational project for CIS5810 - Fall 2025

## Acknowledgments

- EDSR model from [SNU CVLab](https://github.com/sanghyun-son/EDSR-PyTorch)
- Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution" (CVPRW 2017)
