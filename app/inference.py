# app/inference.py
from pathlib import Path
from typing import Dict, Optional
import torch
from torchvision import transforms, models
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Device selection (Apple MPS > CUDA > CPU) ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")        # Apple Silicon GPU
    logger.info("Using MPS device (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")       # NVIDIA GPU
    logger.info("Using CUDA device")
else:
    DEVICE = torch.device("cpu")        # Fallback CPU
    logger.info("Using CPU device")

class InferenceModel:
    """
    Loads a trained EfficientNet-B0 checkpoint and provides a predict() method.
    Expects a checkpoint saved by train.py with keys:
      - "state_dict": model state dict
      - "classes": list of class names (["NORMAL","PNEUMONIA"])
      - "img_size": int (e.g., 224)
    """

    def __init__(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        logger.info(f"Looking for checkpoint at: {ckpt_path.absolute()}")
        
        if not ckpt_path.exists():
            # Try to find the file in common locations
            possible_paths = [
                ckpt_path,
                Path("models") / ckpt_path.name,
                Path("app") / "models" / ckpt_path.name,
                Path("../models") / ckpt_path.name,
                Path(".") / ckpt_path.name,
            ]
            
            found = False
            for path in possible_paths:
                if path.exists():
                    ckpt_path = path
                    found = True
                    logger.info(f"Found checkpoint at: {ckpt_path.absolute()}")
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"Checkpoint not found at: {ckpt_path}\n"
                    f"Tried: {[str(p) for p in possible_paths]}\n"
                    f"Current working directory: {Path.cwd()}"
                )

        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            logger.info(f"Successfully loaded checkpoint from {ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        self.img_size: int = int(ckpt.get("img_size", 224))
        
        # Handle classes with fallback
        if "classes" not in ckpt:
            logger.warning("No classes found in checkpoint, using default ['NORMAL','PNEUMONIA']")
            self.classes = ["NORMAL", "PNEUMONIA"]
        else:
            self.classes = list(ckpt["classes"])
            if len(self.classes) != 2 or not all(c in self.classes for c in ["NORMAL", "PNEUMONIA"]):
                logger.warning(f"Expected classes ['NORMAL','PNEUMONIA'], got {self.classes}. Using default.")
                self.classes = ["NORMAL", "PNEUMONIA"]

        # Build model and load weights
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, len(self.classes))
        
        if "state_dict" in ckpt:
            try:
                self.model.load_state_dict(ckpt["state_dict"])
                logger.info("Successfully loaded model weights")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {e}")
        else:
            logger.warning("No state_dict found in checkpoint. Using untrained model.")

        self.model.eval().to(DEVICE)

        # Preprocessing (must match training normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # cache indices for speed/clarity
        self.idx_normal = self.classes.index("NORMAL")
        self.idx_pneum = self.classes.index("PNEUMONIA")

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
                "status": "success",
                "label": self.classes[label_idx],
                "confidence": float(probs[label_idx]),
                "probs": {c: float(p) for c, p in zip(self.classes, probs)}
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Singleton pattern for model loading
_model_instance = None

def get_model(ckpt_path: Optional[str | Path] = None):
    """Get or create the model instance (singleton pattern)"""
    global _model_instance
    
    if _model_instance is None:
        if ckpt_path is None:
            # Try to find model in default locations
            default_paths = [
                Path("models/best_model.pth"),
                Path("app/models/best_model.pth"),
                Path("../models/best_model.pth"),
                Path("best_model.pth"),
            ]
            
            for path in default_paths:
                if path.exists():
                    ckpt_path = path
                    break
            
            if ckpt_path is None:
                raise FileNotFoundError(
                    "No checkpoint path provided and no default model found.\n"
                    f"Tried: {[str(p) for p in default_paths]}"
                )
        
        _model_instance = InferenceModel(ckpt_path)
    
    return _model_instance
