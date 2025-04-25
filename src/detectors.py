import torch
from torchvision import transforms
from PIL import Image
from src.models import SimpleResNetCNN, AIDetectorResNet
from transformers import AutoModelForImageClassification, AutoProcessor 

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
    def __init__(self, model_path="models/non_poster_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        
    def predict(self, image_path, threshold=0.5):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        confidence = confidence.item()
        return (1 if confidence >= threshold else 0, confidence)