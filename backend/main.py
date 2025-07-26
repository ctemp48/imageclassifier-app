from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io
import urllib.request

app = FastAPI()

# --- Allow frontend calls ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Narrow to ["http://localhost:5500"] for security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model ---
model = models.resnet18(pretrained=True)
model.eval()

# --- Load ImageNet labels ---
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f]

# --- Preprocessing pipeline ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess for ResNet18
    img_t = preprocess(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        _, top_idx = torch.max(probs, 0)

    # Map index to label
    label = imagenet_classes[top_idx.item()]

    return {"prediction": label}
