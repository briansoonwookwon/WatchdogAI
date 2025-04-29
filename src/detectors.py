import torch
from torchvision import transforms
from PIL import Image
from src.models import SimpleResNetCNN, AIDetectorResNet
from transformers import AutoModelForImageClassification, AutoProcessor, ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import mmcv
from mmengine.config import Config
from mmengine.runner import load_state_dict
from mmengine.dataset import pseudo_collate
from mmcv.transforms import Compose
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmengine.logging import HistoryBuffer
import numpy as np
import os

class PosterDetector:
    def __init__(self, model_path="models/SimpleResNetCNN/run_6/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SimpleResNetCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = torch.sigmoid(self.model(tensor))
            
        confidence = output.item()
        return (1 if confidence >= threshold else 0, confidence)

class AI_Poster_Detector:
    def __init__(self, model_path="models/AIDetectorResNet/run_3/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AIDetectorResNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = torch.sigmoid(self.model(tensor))
            
        confidence = output.item()
        return (1 if confidence >= threshold else 0, confidence)

class AI_Non_Poster_Detector:
    def __init__(self, model_path="models/non_poster_model/model.safetensors"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            # Check if the model file exists before attempting to use it
            if os.path.exists(model_path):
                # If we have safetensors module available
                try:
                    from safetensors.torch import load_file
                    
                    # Initialize model with 2 classes (fake/real)
                    self.model = ViTForImageClassification.from_pretrained(
                        "google/vit-large-patch16-224", 
                        num_labels=2,
                        ignore_mismatched_sizes=True
                    ).to(self.device)
                    
                    # Load the weights if safetensors is available
                    state_dict = load_file(model_path)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded custom weights from safetensors file")
                    
                except ImportError:
                    print("Warning: safetensors package not found. Using pretrained model.")
                    raise ImportError("safetensors package not found")
                    
            else:
                # File doesn't exist, raise error to fall back to pretrained
                print(f"Model file {model_path} not found. Using pretrained model.")
                raise FileNotFoundError(f"Model file {model_path} not found")
                
            # Set up the image processor
            self.processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
            self.fallback_mode = False
            
        except Exception as e:
            print(f"Using pretrained model: {e}")
            
            # Skip loading weights and just use pretrained model
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-large-patch16-224", 
                num_labels=2
            ).to(self.device)
            self.model.eval()
            
            # Set up the image processor
            self.processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
            self.fallback_mode = True
        
    def predict(self, image_path, threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get the probability of the FAKE class (class 0)
            fake_prob = probabilities[:, 0].item()
            real_prob = probabilities[:, 1].item()
            
            # If the fake probability is high, predict FAKE
            if fake_prob >= threshold:
                return (0, fake_prob)
            else:
                return (1, real_prob)

class ArtifactDetector:
    def __init__(self, config_file="models/htc_r50_artifact_final/htc_r50_fpn_1x_artifact.py", 
                 checkpoint_file="models/htc_r50_artifact_final/best_coco_bbox_mAP_epoch_11.pth", 
                 detection_threshold=0.15, device='cpu'):
        
        # Register MMDetection modules
        register_all_modules()

        # Load config
        self.cfg = Config.fromfile(config_file)
        self.cfg.model.pop('pretrained', None)

        # Build model
        self.model = MODELS.build(self.cfg.model)
        self.model.eval()

        # Safely allow PyTorch to load necessary pickled objects
        torch.serialization.add_safe_globals([
            HistoryBuffer,
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.float64().dtype.__class__,
            np.int64().dtype.__class__,
        ])

        # Load checkpoint
        ckpt = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        load_state_dict(self.model, state_dict)

        # Save settings
        self.device = device
        self.detection_threshold = detection_threshold

        # 🔧 Move model to device
        self.model.to(device)

        # Build the test pipeline
        self.pipeline = Compose(self.cfg.test_dataloader.dataset.pipeline)

        # Get class names if available
        self.class_names = self.model.dataset_meta['classes'] if hasattr(self.model, 'dataset_meta') else None

    def predict(self, image_path):
        # Load image
        image = mmcv.imread(image_path)

        # Prepare input
        data = dict(img=image, img_path=image_path)
        data = self.pipeline(data)
        data = pseudo_collate([data])

        # Inference
        with torch.no_grad():
            result = self.model.test_step(data)

        # Extract predictions
        pred = result[0]
        bboxes = pred.pred_instances.bboxes.cpu().numpy()
        labels = pred.pred_instances.labels.cpu().numpy()
        scores = pred.pred_instances.scores.cpu().numpy()

        # Check detections
        detections_found = 0
        for bbox, label, score in zip(bboxes, labels, scores):
            if score >= self.detection_threshold:
                detections_found += 1

        # Determine class
        flag = 1 if detections_found > 0 else 0
        return flag
