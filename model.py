import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# Custom Dataset
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
            raise ValueError(f"No images found in {image_dir}. Check the directory structure.")

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Swin Transformer Modeli
class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()  # Remove the original classification head
        self.flatten = nn.Flatten()  # Flatten the output before passing to the classifier
        self.classifier = nn.Linear(in_features * 7 * 7, num_classes)  # Linear layer for classification

    def forward(self, x):
        x = self.swin(x)  # Extract features
        x = self.flatten(x)  # Flatten the feature map
        x = self.classifier(x)  # Apply classifier
        return x


# Eğitim Fonksiyonu
def train_model():
    num_classes = 9
    model = SwinClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(
        image_dir=r"C:\\Users\\ASUS\\OneDrive\\Desktop\\vit_yeni2\\train",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Ensure that the output and label shapes are correct
            if outputs.dim() != 2 or labels.dim() != 1:
                print(f"Unexpected output shape: {outputs.shape}, labels shape: {labels.shape}")
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), 'swin_transformer_model.pth')


if __name__ == "__main__":
    train_model()
