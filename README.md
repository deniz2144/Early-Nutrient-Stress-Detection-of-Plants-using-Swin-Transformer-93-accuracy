## 📁 Dataset

Dataset used:  
**Early Nutrient Stress Detection of Plants**  
📎 [Kaggle Link](https://www.kaggle.com/datasets/raiaone/early-nutrient-stress-detection-of-plants)

The dataset consists of 9 classes:
- Early Nitrogen Stress  
- Early Phosphorus Stress  
- Early Potassium Stress  
- Early Water Stress  
- Early Salt Stress  
- Early Iron Stress  
- Early Zinc Stress  
- Control (Healthy)  
- Others

## 🧠 Model

The model is based on the [Swin Transformer](https://arxiv.org/abs/2103.14030), implemented using PyTorch and the `timm` library.

- Backbone: `swin_base_patch4_window7_224`  
- Output: `Flatten + Linear` (9 classes)  
- Initialized with pretrained ImageNet weights

## ⚙️ Technologies Used

- Python 3  
- PyTorch  
- timm (PyTorch Image Models)  
- scikit-learn (for evaluation metrics)  
- matplotlib & seaborn (for visualization)

## 🔁 Data Preprocessing

**Training-time augmentations:**
- Resize to 224x224  
- Random Horizontal Flip  
- Random Rotation  
- Random Resized Crop  
- Color Jitter  
- Normalization (ImageNet mean/std)

**During testing:**
- Only resizing and normalization applied

## 📊 Training Details

- Epochs: 10  
- Batch Size: 16  
- Optimizer: AdamW  
- Learning Rate: 5e-5  
- Loss Function: CrossEntropyLoss

The trained model is saved in `.pth` format: `swin_transformer_model.pth`

## ✅ Evaluation

The model performance on the test set was measured using the following metrics:
- Accuracy  
- Precision, Recall, F1-score (per class)  
- Confusion Matrix

### To run:

```bash
python model.py   # Training
python test.py    # Testing and evaluation
![WhatsApp Görsel 2025-05-14 saat 14 44 19_d7d5f204](https://github.com/user-attachments/assets/91fc903b-5923-43fb-9b8e-136a91fdd804)



![WhatsApp Görsel 2025-05-14 saat 14 44 13_a65aa9fa](https://github.com/user-attachments/assets/8ab0a108-3b40-4df4-9817-91b69c74a62f)
