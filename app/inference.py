# app/inference.py
from pathlib import Path
from typing import Dict
import torch
from torchvision import transforms, models
from PIL import Image
import os
import sys

# Add the current directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Device selection (Apple MPS > CUDA > CPU) ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")        # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")       # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")        # Fallback CPU

# Try to get the correct path for GitHub deployment
def get_model_path():
    """Find the model file in various possible locations"""
    possible_paths = [
        Path("models/best_model.pth"),
        Path("app/models/best_model.pth"),
        Path("../models/best_model.pth"),
        Path("./models/best_model.pth"),
        Path(os.path.join(os.path.dirname(__file__), "models", "best_model.pth")),
        Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found model at: {path.absolute()}")
            return path
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"Model file 'best_model.pth' not found in any of these locations:\n" +
        "\n".join([f"  - {p.absolute()}" for p in possible_paths]) +
        f"\n\nCurrent working directory: {os.getcwd()}"
    )

class InferenceModel:
    """
    Loads a trained EfficientNet-B0 checkpoint and provides a predict() method.
    Expects a checkpoint saved by train.py with keys:
      - "state_dict": model state dict
      - "classes": list of class names (["NORMAL","PNEUMONIA"])
      - "img_size": int (e.g., 224)
    """

    def __init__(self, ckpt_path: str | Path = None):
        # If no path provided, try to find it automatically
        if ckpt_path is None:
            ckpt_path = get_model_path()
        
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        try:
            # Load checkpoint with appropriate device
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            print(f"✅ Successfully loaded checkpoint from {ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        self.img_size: int = int(ckpt.get("img_size", 224))
        
        # Handle classes gracefully
        if "classes" in ckpt:
            self.classes = list(ckpt["classes"])
            if len(self.classes) != 2:
                print(f"⚠️ Warning: Expected 2 classes, got {len(self.classes)}. Using default.")
                self.classes = ["NORMAL", "PNEUMONIA"]
        else:
            print("⚠️ Warning: No classes found in checkpoint. Using default ['NORMAL','PNEUMONIA']")
            self.classes = ["NORMAL", "PNEUMONIA"]

        # Build model and load weights
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, len(self.classes))
        
        if "state_dict" in ckpt:
            try:
                self.model.load_state_dict(ckpt["state_dict"])
                print("✅ Successfully loaded model weights")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {e}")
        else:
            raise RuntimeError("No state_dict found in checkpoint")

        self.model.eval().to(DEVICE)

        # Preprocessing (must match training normalization)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # cache indices for speed/clarity
        self.idx_normal = self.classes.index("NORMAL") if "NORMAL" in self.classes else 0
        self.idx_pneum = self.classes.index("PNEUMONIA") if "PNEUMONIA" in self.classes else 1
        
        print(f"✅ Model initialized with classes: {self.classes}")
        print(f"✅ Using device: {DEVICE}")

    @torch.inference_mode()
    def predict(self, pil_img: Image.Image, threshold: float = 0.85) -> Dict:
        """
        Predicts NORMAL vs PNEUMONIA with a probability threshold on PNEUMONIA.
        If P(PNEUMONIA) >= threshold -> label=PNEUMONIA else NORMAL.
        Returns: dict with 'label', 'confidence', and 'probs' per class.
        """
        if not isinstance(pil_img, Image.Image):
            raise TypeError("predict expects a PIL.Image.Image")

        try:
            x = self.tf(pil_img).unsqueeze(0).to(DEVICE)

            # Enable mixed precision for faster inference on Apple Silicon
            if DEVICE.type == "mps":
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    logits = self.model(x)
            else:
                logits = self.model(x)

            probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
            pneu_prob = float(probs[self.idx_pneum])
            label_idx = self.idx_pneum if pneu_prob >= threshold else self.idx_normal

            return {
                "label": self.classes[label_idx],
                "confidence": float(probs[label_idx]),
                "probs": {c: float(p) for c, p in zip(self.classes, probs)}
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "probs": {},
                "error": str(e)
            }

# Global instance for reuse
_model_instance = None

def get_model(ckpt_path: str | Path = None):
    """Get or create a singleton model instance"""
    global _model_instance
    if _model_instance is None:
        print("🔄 Loading model for the first time...")
        _model_instance = InferenceModel(ckpt_path)
    return _model_instance
