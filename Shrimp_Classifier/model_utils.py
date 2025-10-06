import os
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Device info akan ditampilkan saat pertama kali load model, bukan saat import

# Constants from CONFIG
NUM_CLASSES = 3
TARGET_SIZE = 224
VAL_RESIZE_SIZE = 256

# Preprocessing configuration
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class mapping with weights and colors
CLASS_MAPPING = [
    {
        'name': 'Healthy', 
        'description': 'Udang sehat tanpa gejala penyakit', 
        'weight': 1.0,
        'color': '#22c55e',  # Green
        'bg_color': '#dcfce7',  # Light green background
        'icon': '✅'
    },
    {
        'name': 'BG', 
        'description': 'Black Gill Disease - Infeksi insang yang menghitam', 
        'weight': 1.5,
        'color': '#f59e0b',  # Orange
        'bg_color': '#fef3c7',  # Light orange background
        'icon': '⚠️'
    },
    {
        'name': 'WSSV', 
        'description': 'White Spot Syndrome Virus - Virus bintik putih', 
        'weight': 1.0,
        'color': '#ef4444',  # Red
        'bg_color': '#fee2e2',  # Light red background
        'icon': '🔴'
    }
]

# Model paths - using raw string to handle Windows paths
MODEL_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Get directory of current file
    'notebooks', 
    'analysis.ipynb',
    'outputs_training_final',
    'models'
))

MODEL_PATHS = [
    os.path.join(MODEL_BASE_PATH, f'best_model_fold_{i}_20250608_152646.pth')
    for i in range(1, 6)
]

# Path verification akan dilakukan saat load model, bukan saat import

# Create transform pipeline exactly like training
transform = A.Compose([
    A.Resize(VAL_RESIZE_SIZE, VAL_RESIZE_SIZE, interpolation=cv2.INTER_AREA),
    A.CenterCrop(TARGET_SIZE, TARGET_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

class ShrimpClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ShrimpClassifier, self).__init__()
        # Removed print statements for cleaner output
        try:
            # Using the exact same model name from CONFIG
            self.model = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k', 
                                         pretrained=False,  # Set to False since we're loading trained weights
                                         num_classes=num_classes)
        except Exception as e:
            print(f"❌ Error creating model architecture: {str(e)}")
            raise
    
    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict):
        """Custom load_state_dict to handle model prefix"""
        try:
            # Try loading directly first
            super().load_state_dict(state_dict)
        except:
            # If that fails, try to load into the model attribute directly
            self.model.load_state_dict(state_dict)

def load_ensemble_models():
    """Load all ensemble models"""
    print("\n" + "="*50)
    print("🚀 STARTING ENSEMBLE MODEL LOADING")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    if device.type == 'cuda':
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    print(f"📂 Model path: {MODEL_BASE_PATH}")
    
    models = []
    
    for i, model_path in enumerate(MODEL_PATHS, 1):
        print(f"\n🔄 Load model {i}/5")
        print(f"   📁 {os.path.basename(model_path)}")
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"   🏗️  Creating model architecture...")
            model = ShrimpClassifier(NUM_CLASSES)
            
            print(f"   📥 Loading state dict...")
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            
            print(f"   🔗 Loading weights into model...")
            model.load_state_dict(state_dict)
            
            model.eval()
            model = model.to(device)
            models.append(model)
            
            print(f"   ✅ Model {i} loaded successfully!")
            
        except Exception as e:
            print(f"   ❌ Error loading model {i}: {str(e)}")
            raise
    
    print(f"\n" + "="*50)
    print(f"🎉 SUCCESS: All {len(models)} models loaded!")
    print("="*50)
    return models

def preprocess_image(image_path):
    """Preprocess image for model input using the same transforms as training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read image using cv2 to match training
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(device)

def get_ensemble_predictions(image_tensor, models):
    """Get predictions from all models"""
    all_outputs = []
    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            all_outputs.append(probabilities)
    return all_outputs

def majority_voting(all_probabilities):
    """Implement majority voting ensemble method"""
    predictions = [torch.argmax(prob).item() for prob in all_probabilities]
    unique, counts = np.unique(predictions, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    confidence = np.max(counts) / len(predictions)
    return majority_class, confidence

def averaging(all_probabilities, weights=None):
    """
    Implement averaging ensemble method with optional weights
    Args:
        all_probabilities: List of probability tensors
        weights: Optional list of weights per class. If None, uses equal weights.
    """
    if weights is None:
        # Simple averaging
        avg_probs = torch.stack(all_probabilities).mean(dim=0)
    else:
        # Weighted averaging
        weights_tensor = torch.tensor(weights, device=all_probabilities[0].device)
        weighted_probs = torch.stack([prob * weights_tensor for prob in all_probabilities])
        avg_probs = weighted_probs.mean(dim=0)
    
    predicted_class_tensor = torch.argmax(avg_probs)
    predicted_class = predicted_class_tensor.item()
    confidence = avg_probs[predicted_class_tensor].item()
    return predicted_class, confidence, avg_probs.cpu().numpy()

def predict_ensemble(image_path, models, method='weighted_averaging'):
    """
    Melakukan prediksi ensemble pada gambar
    
    Args:
        image_path: Path ke file gambar
        models: List model yang akan digunakan untuk ensemble
        method: Metode ensemble ('weighted_averaging', 'majority_voting', 'simple_averaging')
        
    Returns:
        Dict hasil prediksi ensemble dengan metode terpilih sebagai hasil utama
    """
    try:
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        
        # Get predictions from all models
        all_probabilities = []
        individual_predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze()  # Remove batch dimension
                all_probabilities.append(probabilities)
                
                # Get individual prediction for this model
                pred_class_tensor = torch.argmax(probabilities)
                pred_class = pred_class_tensor.item()
                individual_predictions.append({
                    'predicted_class': pred_class,
                    'predicted_class_name': CLASS_MAPPING[pred_class]['name'],
                    'confidence': float(probabilities[pred_class_tensor]),
                    'probabilities': probabilities.cpu().tolist()
                })
        
        # Stack all probabilities for ensemble calculations
        stacked_probs = torch.stack(all_probabilities)  # Shape: (num_models, num_classes)
        
        # === 1. MAJORITY VOTING ===
        pred_classes = [torch.argmax(prob).item() for prob in all_probabilities]
        unique_classes, counts = np.unique(pred_classes, return_counts=True)
        
        # Create vote counts for all classes
        vote_counts = np.zeros(len(CLASS_MAPPING))
        for cls, count in zip(unique_classes, counts):
            vote_counts[cls] = count
            
        majority_class = np.argmax(vote_counts)
        majority_confidence = vote_counts[majority_class] / len(models)
        majority_votes_dict = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
        
        # Convert votes to probabilities (for percentage display)
        majority_probs = vote_counts / len(models)
        
        # Create detailed voting count for display
        voting_details = []
        for i, class_info in enumerate(CLASS_MAPPING):
            vote_count = int(vote_counts[i])
            percentage = (vote_count / len(models)) * 100
            voting_details.append({
                'class_name': class_info['name'],
                'class_index': i,
                'vote_count': vote_count,
                'percentage': percentage,
                'display_text': f"{class_info['name']}: {vote_count} vote ({percentage:.0f}%)"
            })
        
        # === 2. SIMPLE AVERAGING ===
        simple_avg_probs = torch.mean(stacked_probs, dim=0)  # Average across models
        simple_pred_tensor = torch.argmax(simple_avg_probs)
        simple_pred = simple_pred_tensor.item()
        simple_confidence = float(simple_avg_probs[simple_pred_tensor])
        
        # === 3. WEIGHTED AVERAGING ===
        # Get class weights from CLASS_MAPPING
        class_weights = torch.tensor([c['weight'] for c in CLASS_MAPPING], device=stacked_probs.device)
        
        # Apply weights to the averaged probabilities
        weighted_avg_probs = simple_avg_probs * class_weights
        # Normalize to make it a proper probability distribution
        weighted_avg_probs = weighted_avg_probs / weighted_avg_probs.sum()
        
        weighted_pred_tensor = torch.argmax(weighted_avg_probs)
        weighted_pred = weighted_pred_tensor.item()
        weighted_confidence = float(weighted_avg_probs[weighted_pred_tensor])
        
        # === ENSEMBLE RESULTS ===
        ensemble_results = {
            'majority_voting': {
                'predicted_class': int(majority_class),
                'predicted_class_name': CLASS_MAPPING[majority_class]['name'],
                'confidence': float(majority_confidence),
                'votes': majority_votes_dict,
                'vote_distribution': vote_counts.tolist(),
                'voting_details': voting_details,  # Detailed voting count for display
                'probabilities': majority_probs.tolist()  # Add probabilities for percentage display
            },
            'simple_averaging': {
                'predicted_class': int(simple_pred),
                'predicted_class_name': CLASS_MAPPING[simple_pred]['name'],
                'confidence': simple_confidence,
                'probabilities': simple_avg_probs.cpu().tolist()
            },
            'weighted_averaging': {
                'predicted_class': int(weighted_pred),
                'predicted_class_name': CLASS_MAPPING[weighted_pred]['name'],
                'confidence': weighted_confidence,
                'probabilities': weighted_avg_probs.cpu().tolist(),
                'class_weights_used': [c['weight'] for c in CLASS_MAPPING]
            }
        }
        
        # === SELECT METHOD AS MAIN RESULT ===
        if method not in ensemble_results:
            print(f"Warning: Unknown method '{method}'. Using 'weighted_averaging' as default.")
            method = 'weighted_averaging'
            
        selected_result = ensemble_results[method]
        
        # Create distribution details with colors for display
        probabilities = selected_result.get('probabilities', [])
        distribution_details = []
        for i, class_info in enumerate(CLASS_MAPPING):
            probability = probabilities[i] if i < len(probabilities) else 0.0
            percentage = probability * 100
            distribution_details.append({
                'class_name': class_info['name'],
                'class_index': i,
                'probability': probability,
                'percentage': percentage,
                'color': class_info['color'],
                'bg_color': class_info['bg_color'],
                'icon': class_info['icon'],
                'description': class_info['description'],
                'is_predicted': i == selected_result['predicted_class']
            })
        
        return {
            'input_image': image_path,
            'individual_model_predictions': individual_predictions,
            # === MAIN RESULT (based on selected method) ===
            'predicted_class': selected_result['predicted_class'],
            'predicted_class_name': selected_result['predicted_class_name'],
            'predicted_class_index': selected_result['predicted_class'],  # For backward compatibility
            'confidence': selected_result['confidence'],
            'probabilities': selected_result.get('probabilities', []),  # Use majority voting probabilities for bars
            'distribution_details': distribution_details,  # Detailed distribution with colors
            'voting_details': selected_result.get('voting_details', []),  # Detailed voting count
            'method_used': method,
            # === ALL ENSEMBLE METHODS (for comparison) ===
            'ensemble_methods': ensemble_results,
            # === METADATA ===
            'num_models_used': len(models),
            'class_mapping': CLASS_MAPPING
        }
        
    except Exception as e:
        print(f"Error in predict_ensemble: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 