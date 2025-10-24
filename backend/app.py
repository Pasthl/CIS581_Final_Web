import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
from datetime import datetime

from src import get_model
from src.preprocessing import preprocess_pipeline_custom

# Configuration
UPLOAD_FOLDER = 'storage/uploads'
OUTPUT_FOLDER = 'storage/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for frontend
CORS(app)

# Ensure storage folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load EDSR model (lazy loading)
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_edsr_model():
    """Get or initialize EDSR model"""
    global model
    if model is None:
        model_path = os.path.join('models', 'edsr_baseline_x4-6b446fab.pt')
        print("Initializing EDSR model...")
        model = get_model(model_path=model_path, scale=4, device='cpu')
        print("Model ready!")
    return model


@app.route('/')
def index():
    """API health check"""
    return jsonify({
        'status': 'ok',
        'message': 'Pixel Revival API is running',
        'version': '1.0.0'
    })


@app.route('/api/denoise', methods=['POST'])
def denoise_image():
    """
    Process image with EDSR denoising

    Expected: multipart/form-data with 'image' file
    Returns: JSON with processed image URL
    """
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input.{file_ext}"
        output_filename = f"{unique_id}_output.png"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Save uploaded file
        file.save(input_path)
        print(f"Received image: {input_filename}")

        # Run EDSR inference (without saving to file)
        print("Running EDSR inference...")
        start_time = datetime.now()

        edsr_model = get_edsr_model()
        # Use infer method without output_path to avoid saving
        result_image = edsr_model.infer(input_path, output_path=None)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Processing completed in {elapsed:.2f}s")

        # Get image dimensions (use with statement to ensure file is closed)
        with Image.open(input_path) as input_img:
            input_size = input_img.size
        output_size = result_image.size

        # Convert output image to base64 (so we don't need to save it)
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Clean up input file immediately (file is now closed)
        try:
            os.remove(input_path)
            print(f"[OK] Cleaned up temporary files (no output file created)")
        except Exception as cleanup_error:
            print(f"[WARN] Failed to cleanup {input_path}: {cleanup_error}")

        # Return result with base64 image data
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'input': {
                'filename': file.filename,
                'size': input_size,
            },
            'output': {
                'size': output_size,
                'image_data': f'data:image/png;base64,{img_base64}'  # Base64 data URL
            },
            'processing_time': f"{elapsed:.2f}s"
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Try to clean up on error too
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                os.remove(input_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass
        return jsonify({
            'error': 'Failed to process image',
            'details': str(e)
        }), 500


@app.route('/api/pipeline', methods=['POST'])
def process_pipeline():
    """
    Flexible processing pipeline with step selection
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type'}), 400

        # Get step toggles
        enable_preprocess = request.form.get('enable_preprocess', 'true').lower() == 'true'
        enable_deblur = request.form.get('enable_deblur', 'true').lower() == 'true'
        enable_edsr = request.form.get('enable_edsr', 'true').lower() == 'true'

        # Save uploaded file
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input.{file_ext}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        print(f"Processing pipeline for: {input_filename}")
        print(f"  Steps: Preprocess={enable_preprocess}, Deblur={enable_deblur}, EDSR={enable_edsr}")
        start_time = datetime.now()

        # Load original image
        current_img = Image.open(input_path)
        result = {}

        # Step 1: Preprocessing
        if enable_preprocess:
            print("  [1] Preprocessing...")
            preprocessed_img = preprocess_pipeline_custom(
                input_path,
                None,
                remove_artifacts=False,
                enhance_contrast=True,
                contrast_method='clahe',
                contrast_clip=1.5,
                denoise=False,
                gamma=None
            )
            current_img = preprocessed_img
            result['preprocessed'] = preprocessed_img
        else:
            print("  [1] Preprocessing skipped")
            result['preprocessed'] = None

        # Step 2: Deblur (placeholder)
        if enable_deblur:
            print("  [2] Deblur (placeholder)...")
            deblurred_img = current_img
            current_img = deblurred_img
            result['deblurred'] = deblurred_img
        else:
            print("  [2] Deblur skipped")
            result['deblurred'] = None

        # Step 3: EDSR Super-Resolution
        if enable_edsr:
            print("  [3] EDSR Super-Resolution...")
            edsr_model = get_edsr_model()
            edsr_img = edsr_model.infer_from_pil(current_img, output_path=None)
            result['edsr'] = edsr_img
        else:
            print("  [3] EDSR skipped")
            result['edsr'] = None

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Pipeline completed in {elapsed:.2f}s")

        # Convert to base64
        def img_to_base64(img):
            if img is None:
                return None
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode("utf-8")}'

        # Clean up
        try:
            os.remove(input_path)
        except Exception as e:
            print(f"Cleanup warning: {e}")

        return jsonify({
            'success': True,
            'message': 'Pipeline completed',
            'preprocessed': img_to_base64(result.get('preprocessed')),
            'deblurred': img_to_base64(result.get('deblurred')),
            'edsr': img_to_base64(result.get('edsr')),
            'processing_time': f"{elapsed:.2f}s"
        })

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<filename>')
def serve_image(filename):
    """Serve processed or uploaded images"""
    try:
        # Check upload folder first
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(upload_path):
            return send_file(upload_path, mimetype='image/png')

        # Check output folder
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(output_path):
            return send_file(output_path, mimetype='image/png')

        return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_files():
    """Clean up old uploaded/processed files (optional maintenance endpoint)"""
    try:
        import time
        from pathlib import Path

        # Delete files older than 1 hour
        current_time = time.time()
        max_age = 3600  # 1 hour in seconds

        deleted_count = 0
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for filepath in Path(folder).glob('*'):
                if filepath.is_file():
                    file_age = current_time - filepath.stat().st_mtime
                    if file_age > max_age:
                        filepath.unlink()
                        deleted_count += 1

        return jsonify({
            'success': True,
            'message': f'Cleaned up {deleted_count} old files'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Pixel Revival Backend API")
    print("=" * 50)
    print("Starting server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
