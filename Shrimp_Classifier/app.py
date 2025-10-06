import os
import json
import uuid
import base64
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename
from model_utils import load_ensemble_models, predict_ensemble, CLASS_MAPPING

# Konfigurasi aplikasi
app = Flask(__name__)
app.secret_key = 'udangClassifierSecretKey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Pastikan folder uploads tersedia
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable untuk menyimpan model ensemble
ensemble_models = None
models_loading_attempted = False

def get_ensemble_models():
    """
    Lazy loading untuk ensemble models - hanya memuat ketika dibutuhkan
    """
    global ensemble_models, models_loading_attempted
    
    if ensemble_models is not None:
        # Model sudah dimuat sebelumnya
        return ensemble_models
    
    if models_loading_attempted:
        # Sudah pernah mencoba load tapi gagal
        return None
    
    # Tandai bahwa kita sudah mencoba loading
    models_loading_attempted = True
    
    print("Memuat model ensemble...")
    try:
        ensemble_models = load_ensemble_models()
        print(f"Berhasil memuat {len(ensemble_models)} model ensemble")
        return ensemble_models
    except Exception as e:
        print(f"Error memuat model ensemble: {str(e)}")
        print("Aplikasi akan dijalankan tanpa model. Silakan periksa file model.")
        ensemble_models = None
        return None

# LAZY LOADING - Model dimuat hanya saat request pertama (menghindari duplicate loading)
# Uncomment baris berikut untuk eager loading:
# print("🚀 Loading models at startup...")
# ensemble_models = get_ensemble_models()

def allowed_file(filename):
    """
    Memeriksa apakah file yang diupload memiliki ekstensi yang diizinkan
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """
    Menyimpan file yang diupload ke disk dan mengembalikan path-nya
    """
    # Buat nama file unik berdasarkan timestamp dan UUID
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_filename = f"{timestamp}-{uuid.uuid4().hex[:8]}_{filename}"
    
    # Simpan file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    return file_path

def encode_image(file):
    """
    Encode uploaded image to base64 for displaying in HTML
    """
    # Read the file content
    file_content = file.read()
    # Reset file pointer to beginning
    file.seek(0)
    # Encode to base64
    encoded = base64.b64encode(file_content).decode('utf-8')
    return encoded

@app.route('/')
def index():
    """Tampilan utama web interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file_web():
    """
    Endpoint untuk web interface
    """
    # Cek apakah model telah dimuat
    models = get_ensemble_models()
    if models is None:
        flash('Model belum dimuat. Silakan cek konfigurasi server.', 'error')
        return redirect(url_for('index'))
    
    # Cek apakah ada file dalam request
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Cek apakah file kosong
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))
    
    # Cek apakah file valid
    if file and allowed_file(file.filename):
        try:
            # Simpan file dengan nama unik
            if file.filename:
                safe_filename = secure_filename(file.filename)
            else:
                safe_filename = 'uploaded_image.jpg'
            filename = str(uuid.uuid4()) + '_' + safe_filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Pastikan direktori upload ada
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Simpan file
            file.save(file_path)
            
            # Prediksi menggunakan ensemble model - UBAH KE MAJORITY VOTING
            # result = predict_ensemble(file_path, models, method='weighted_averaging')  # BACKUP: Weighted averaging (original)
            result = predict_ensemble(file_path, models, method='majority_voting')  # MAIN: Use Majority Voting as primary method
            
            return render_template('result.html', 
                                image_filename=filename,
                                result=result,
                                class_mapping=CLASS_MAPPING)
            
        except Exception as e:
            # Jika terjadi error, hapus file jika ada
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            flash(f'Error dalam memproses gambar: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Format file tidak diizinkan', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for image prediction with optimized request processing"""
    # Thread-local request validation dengan efficient error propagation
    models = get_ensemble_models()
    if models is None:
        return jsonify({
            'success': False,
            'error': 'Model belum dimuat. Silakan cek konfigurasi server.',
            'error_code': 'MODEL_NOT_LOADED'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Tidak ada file yang dikirim',
            'error_code': 'NO_FILE'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Tidak ada file yang dipilih',
            'error_code': 'EMPTY_FILENAME'
        }), 400
    
    # Validation with early exit pattern
    if not file or not allowed_file(file.filename):
        return jsonify({
            'success': False,   
            'error': 'Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau WebP',
            'error_code': 'INVALID_FORMAT'
        }), 400
    
    try:
        # Optimized file saving dengan buffer pre-allocation
        file_path = save_uploaded_file(file)
        
        # Process image file
        print(f"Processing image: {file_path}")
        
        # Make prediction dengan enhanced error context
        try:
            # result = predict_ensemble(file_path, models, method='weighted_averaging')  # BACKUP: Weighted averaging (original)
            result = predict_ensemble(file_path, models, method='majority_voting')  # MAIN: Use Majority Voting as primary method
            
            # Optimize result data structure untuk minimal JSON serialization overhead
            response = {
                'success': True,
                'filename': os.path.basename(file_path),
                'result': result
            }
            
            # Add conversion information jika dilakukan
            if 'file_metadata' in result:
                metadata = result.pop('file_metadata')  # Remove from result
                
                # Refine conversion info dan state management
                if metadata.get('conversion_applied'):
                    response['conversion'] = {
                        'original_format': metadata.get('original_format'),
                        'converted_format': metadata.get('processed_format'),
                        'converted': True
                    }
                    response['note'] = f"Format {metadata.get('original_format', 'WebP')} telah dikonversi ke {metadata.get('processed_format', 'JPG')} untuk kompatibilitas."
            
            return jsonify(response)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Prediction error: {error_msg}")
            
            # Error handling untuk prediction
            return jsonify({
                'success': False,
                'error': f'Prediction error: {error_msg}',
                'error_code': 'PREDICTION_ERROR'
            }), 500
                
    except Exception as e:
        error_msg = str(e)
        print(f"Processing error: {error_msg}")
        
        # Error classification
        status_code = 500
        error_code = 'PROCESSING_ERROR'
        
        if "tidak ditemukan" in error_msg.lower():
            error_code = 'FILE_NOT_FOUND'
            status_code = 404
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': error_code
        }), status_code

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Menampilkan file yang sudah diupload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handler untuk error file terlalu besar"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'File terlalu besar. Maksimal 16MB'
        }), 413
    else:
        flash('File terlalu besar. Maksimal 16MB', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handler untuk error halaman tidak ditemukan"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Endpoint tidak ditemukan'
        }), 404
    else:
        return render_template('404.html'), 404

if __name__ == '__main__':
    # Jalankan aplikasi
    print("Menjalankan aplikasi UdangClassifier...")
    app.run(debug=True, host='0.0.0.0', port=5000) 