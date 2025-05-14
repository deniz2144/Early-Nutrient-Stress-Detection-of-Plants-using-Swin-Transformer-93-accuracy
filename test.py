import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import timm
from torch import nn
import numpy as np

# Dataset class (aynı)
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transform

        for idx, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_path = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(class_path, fname)
                    self.image_paths.append(full_path)
                    self.labels.append(idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Model tanımı
class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features * 7 * 7, num_classes)

    def forward(self, x):
        x = self.swin(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Değerlendirme fonksiyonu
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 9

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dir = r"C:\\Users\\ASUS\\OneDrive\\Desktop\\vit_yeni2\\test"
    test_dataset = CustomDataset(image_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = sorted(os.listdir(test_dir))

    # Model yükleniyor
    model = SwinClassifier(num_classes)
    model.load_state_dict(torch.load("swin_transformer_model.pth", map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Konfüzyon matrisi çizimi
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Precision, Recall, F1-score
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_model()
