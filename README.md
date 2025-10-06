# UdangClassifier - Sistem Klasifikasi Penyakit Udang

<table align="center">
  <tr>
    <!-- Kolom kiri: gambar besar -->
    <td>
      <img src="https://github.com/Affand6331/Shrimp_Classifier/blob/main/asset/image1.png" width="500">
    </td>
    <!-- Kolom kanan: 3 gambar vertikal -->
    <td>
      <img src="https://github.com/Affand6331/Shrimp_Classifier/blob/main/asset/image2.png" width="250"><br>
      <img src="https://github.com/Affand6331/Shrimp_Classifier/blob/main/asset/image3.png" width="250"><br>
      <img src="https://github.com/Affand6331/Shrimp_Classifier/blob/main/asset/image4.png" width="250">
    </td>
  </tr>
</table>


Sistem klasifikasi penyakit udang berbasis deep learning menggunakan arsitektur Swin Transformer dengan metode ensemble. Sistem ini dapat mengklasifikasikan gambar udang ke dalam 3 kategori:

- **Healthy** (Udang sehat)
- **BG** (Black Gill Disease - Penyakit insang hitam)  
- **WSSV** (White Spot Syndrome Virus - Virus bintik putih)

## Fitur Utama

- 🔬 **Model Ensemble**: Menggunakan 5 model Swin Transformer untuk akurasi tinggi
- 🌐 **Web Interface**: Interface web yang modern dan mudah digunakan
- 📡 **API Endpoint**: RESTful API untuk integrasi dengan aplikasi lain
- 📸 **Upload Gambar**: Upload file gambar atau gunakan kamera langsung
- 🎯 **Multiple Ensemble Methods**: Simple averaging, weighted averaging, dan majority voting
- ⚡ **GPU Support**: Dukungan CUDA untuk inferensi lebih cepat

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- CUDA-capable GPU (opsional, untuk inferensi lebih cepat)
- Webcam (opsional, untuk pengambilan gambar langsung)
- Minimum 4GB RAM (8GB direkomendasikan)

## Instalasi

1. Buat virtual environment (disarankan):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask pillow opencv-python albumentations timm numpy
```

3. Siapkan model:
   - Letakkan model ensemble (`.pth` files) di folder:
   - `notebooks/analysis.ipynb/outputs_training_final/models/`
   - Model yang dibutuhkan: `best_model_fold_1_20250608_152646.pth` hingga `best_model_fold_5_20250608_152646.pth`

4. Pastikan folder upload tersedia:
```bash
mkdir -p uploads
```

## Penggunaan

### Web Interface

1. Jalankan aplikasi web:
```bash
cd Shrimp_Classifier
python app.py
```

2. Buka browser dan akses `http://localhost:5000`

3. Upload gambar udang atau gunakan kamera untuk mengambil gambar

4. Sistem akan menampilkan hasil klasifikasi beserta probabilitas untuk setiap kelas

### API Endpoint

Untuk integrasi programatis, gunakan API endpoint:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/api/predict
```

Response JSON:
```json
{
  "success": true,
  "result": {
    "predicted_class_name": "Healthy",
    "predicted_class_index": 0,
    "probabilities": [0.85, 0.10, 0.05],
    "ensemble_methods": {
      "weighted_averaging": {...},
      "simple_averaging": {...},
      "majority_voting": {...}
    }
  }
}
```

## Struktur Folder

```
Shrimp_Classifier/
├── app.py                          # Aplikasi Flask utama
├── model_utils.py                  # Utilities untuk model dan prediksi
├── notebooks/
│   └── analysis.ipynb/
│       └── outputs_training_final/
│           └── models/             # Model ensemble (.pth files)
├── static/
│   └── css/
│       └── style.css              # CSS styling
├── templates/
│   ├── index.html                 # Halaman utama
│   ├── result.html                # Halaman hasil klasifikasi
│   └── 404.html                   # Halaman error 404
├── uploads/                       # Folder untuk gambar yang diupload
├── data/                          # Dataset (jika ada)
├── results/                       # Hasil eksperimen
└── Output_Hasil/                  # Output hasil training
```

## Teknologi yang Digunakan

### Backend
- **Flask**: Web framework untuk API dan web interface
- **PyTorch**: Deep learning framework
- **Swin Transformer**: Arsitektur model utama (timm library)
- **Albumentations**: Image augmentation dan preprocessing
- **OpenCV**: Computer vision operations

### Frontend
- **HTML5 & CSS3**: Modern web interface
- **Bootstrap Icons**: Icon library
- **Inter Font**: Typography
- **Responsive Design**: Support untuk desktop dan mobile

### Model Architecture
- **Swin Transformer Tiny**: Pre-trained pada ImageNet-22k
- **Ensemble Learning**: 5-fold cross validation models
- **Multiple Ensemble Methods**: Averaging dan voting strategies
- **Class Weighting**: Balanced prediction untuk dataset imbalanced

## Metode Ensemble

Sistem menggunakan 3 metode ensemble yang berbeda:

1. **Simple Averaging**: Rata-rata probabilitas dari semua model
2. **Weighted Averaging**: Rata-rata dengan bobot berdasarkan kelas (default)
3. **Majority Voting**: Voting berdasarkan prediksi mayoritas

## Format File yang Didukung

- JPG/JPEG
- PNG
- Maksimal ukuran file: 16MB

## Troubleshooting

### Model tidak ditemukan
Pastikan file model berada di lokasi yang benar:
```
notebooks/analysis.ipynb/outputs_training_final/models/
```

### Error CUDA
Jika menggunakan CPU, sistem akan secara otomatis fallback ke CPU inference.

### Error memory
Kurangi batch size atau gunakan model dengan resolusi lebih kecil.

## Performance

- **Akurasi**: ~95% pada test set
- **Inference Time**: 
  - GPU: ~100ms per gambar
  - CPU: ~1-2 detik per gambar
- **Model Size**: ~90MB per model (450MB total untuk ensemble)

## Lisensi

MIT License

