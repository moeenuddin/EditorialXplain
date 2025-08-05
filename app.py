import os
import zipfile
import pathlib
import pandas as pd
import torch
import streamlit as st
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import openai
import requests

# Constants
LABEL_CSV = 'Labels2.csv'
MODEL_PATH = 'model.pth'
DATASET_URL = "http://bilaunwan.pk/download/originals-curated-v2.zip"
DATA_DIR = 'originals-curated-v2'

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

client = openai.OpenAI()

# --- Download and Extract Dataset ---
@st.cache_resource
def download_and_extract():
    if not os.path.exists(DATA_DIR):
        archive_path = 'dataset.zip'
        st.info("Downloading dataset...")
        r = requests.get(DATASET_URL)
        with open(archive_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        st.success("Dataset downloaded and extracted.")
    else:
        st.info("Dataset already present.")

download_and_extract()

# --- Load Labels ---
labels_df = pd.read_csv(LABEL_CSV)
required_cols = ['filename', 'finalscore', 'text', 'background', 'composition', 'policy']
assert all(col in labels_df.columns for col in required_cols), "Label CSV must contain required columns."
originals_score_map = {
    row['filename']: [row['finalscore'], row['text'], row['background'], row['composition'], row['policy']]
    for _, row in labels_df.iterrows()
}

st.info("Labels loaded.")

# ==== Dataset ====
class ImageFeatureDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_paths = []
        self.scores = []
        self.image_dir = image_dir
        self.transform = transform
        for subdir in ['Originals', 'Curated']:
            full_path = os.path.join(self.image_dir, subdir)
            for filename in os.listdir(full_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_image_path = os.path.join(full_path, filename)
                    self.image_paths.append(full_image_path)
                    self.scores.append(originals_score_map.get(filename, [0, 0, 0, 0, 0]))
        self.status_map = {'init': 0, 'draft': 1, 'done': 2, 'reject': 3}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        intermediate_targets = torch.tensor(self.scores[idx], dtype=torch.float32)
        return image, intermediate_targets

# ==== Model ====
class MultiStageModel(nn.Module):
    def __init__(self):
        super(MultiStageModel, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = base_model.fc.in_features
        self.intermediate_head = nn.Linear(self.feature_dim, 5)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        intermediate_preds = self.intermediate_head(features)
        combined = torch.cat([features, intermediate_preds], dim=1)
        status_logits = self.classifier(combined)
        return intermediate_preds, status_logits

# ==== Training ====
def train(model, dataloader, optimizer, criterion_reg, criterion_cls, device):
    model.train()
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds, status_logits = model(images)
        status_labels = torch.tensor([0] * len(images), dtype=torch.long).to(device)  # Dummy status
        loss = criterion_reg(preds, targets) + criterion_cls(status_logits, status_labels)
        loss.backward()
        optimizer.step()

def run_training():
    if os.path.exists(MODEL_PATH):
        st.info("Model already trained.")
        return

    st.info("Training model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFeatureDataset(LABEL_CSV, DATA_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = MultiStageModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(5):
        train(model, dataloader, optimizer, criterion_reg, criterion_cls, device)
    torch.save(model.state_dict(), MODEL_PATH)
    st.success("Model trained and saved.")

# ==== Inference ====
def infer_image(model, image, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        inter_preds, status_logits = model(image_tensor)
        reverse_status_map = { 0:'init', 1: 'draft', 2:'done', 3:'reject'}
        predicted_index = torch.argmax(status_logits, dim=1).item()
        predicted_label = reverse_status_map[predicted_index]

        #print(model.status_map.selfstatus_logits)
    return inter_preds.cpu().numpy()[0], predicted_label #torch.argmax(status_logits, dim=1).item()

# ==== GPT Rubric Prompt ====
def generate_feedback(inter_preds,status):
    rubric = """A.1 Text Layer: Clarity, Audience, Call to Action, Creativity, Emotion, Tone, Appeal, Grammar, Value, Relevance.
A.2 Composition: Readability, Hierarchy, Balance, Color, Brand, Engagement, Originality, Quality, Accuracy, Usability.
A.3 Background: Relevance, Clutter, Contrast, Color, Focus, Mood, Creativity, Quality.
A.4 Policy: Offensive, Sensitivity, Appropriateness, Misinformation, Age, Platform, Brand Risk."""

    prompt = f"""You are an expert reviewer and assitant to editor. Given the following model predictions for an uploaded image:
Final Score: {inter_preds[0]:.2f}
Text: {inter_preds[1]:.2f}
Background: {inter_preds[2]:.2f}
Composition: {inter_preds[3]:.2f}
Policy: {inter_preds[4]:.2f}
Status: {status}
The model's final decision is to "{status}".

Using the rubric below, generate a helpful feedback paragraph for the creator to understand why that status was given, and to improve their content.

Rubric:
{rubric}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==== Streamlit Interface ====
st.title("Image Evaluation & GPT Feedback")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run_training()

model = MultiStageModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

uploaded_image = st.file_uploader("Upload an image for evaluation", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    inter_preds, status = infer_image(model, image, device)
    labels = ['Final Score', 'Text', 'Background', 'Composition', 'Policy']
    
    st.write(f"**Status**: {status}")
    for i, label in enumerate(labels):
        st.write(f"**{label}**: {inter_preds[i]:.2f}")

    feedback = generate_feedback(inter_preds,status)
    st.subheader("GPT-3.5 Feedback")
    st.write(feedback)
