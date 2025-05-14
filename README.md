
# 🌿 Early Detection of Plant Nutrient Stress with Swin Transformer

This project implements a deep learning-based computer vision system that automatically detects early nutrient stress in plant leaves using Swin Transformer architecture. It is designed as a practical solution that could be integrated into smart agriculture systems for proactive crop monitoring.

---

## 📁 Dataset

For this project, I worked with the **Early Nutrient Stress Detection of Plants** dataset, available on Kaggle:  
📎 [Kaggle Link](https://www.kaggle.com/datasets/raiaone/early-nutrient-stress-detection-of-plants)

The dataset is designed for multi-class image classification and contains thousands of annotated leaf images categorized into 9 classes:

-ashgourd_fresh
-ashgourd_nitrogen
-ashgourd_potassium
-bittergourd_fresh
-bittergourd_nitrogen
-bittergourd_potassium
-snakegourd_fresh
-snakegourd_nitrogen
-snakegourd_potassium

What makes this dataset particularly interesting is the subtlety of the differences between stress conditions — which is ideal for evaluating the ability of deep learning models to detect fine-grained visual cues in a biological context. Unlike synthetic or highly-contrasted datasets, this one simulates real agricultural conditions, which I find both more challenging and more rewarding to work with.

---

## 🧠 Model

I chose to use the [Swin Transformer](https://arxiv.org/abs/2103.14030) for this project, implemented via PyTorch and the `timm` library. My reasoning was simple: Swin Transformer has proven to outperform many CNN-based models on fine-grained classification tasks due to its ability to capture both local and global dependencies effectively.

### 🔧 Architecture Details:

- **Backbone:** `swin_base_patch4_window7_224`  
- **Classification Head:** Replaced with a custom `Flatten + Linear` layer adapted for 9 output classes  
- **Weights:** Initialized using pretrained ImageNet parameters for transfer learning efficiency

By replacing the final classification head and fine-tuning the feature extractor on my domain-specific dataset, I was able to achieve high performance with relatively limited training time.

---

## ⚙️ Technologies Used

This project was built using the following stack:

- **Python 3** – for flexibility and rapid prototyping  
- **PyTorch** – core deep learning framework  
- **timm** – for easy access to SOTA pretrained transformer models  
- **scikit-learn** – to compute detailed evaluation metrics  
- **matplotlib & seaborn** – for high-quality visualizations

Everything was built with reproducibility and readability in mind.

---

## 🔁 Data Preprocessing

Having worked on vision systems before, I know how critical preprocessing and data augmentation can be — especially with datasets derived from real environments like farms or greenhouses.

So I designed a data pipeline that both standardizes the image input and introduces robust variability for generalization. Here's what it looks like during **training**:

- 🔄 `Resize` to 224×224  
- ↔️ `Random Horizontal Flip`  
- 🔁 `Random Rotation` (±15°)  
- 🔍 `Random Resized Crop`  
- 🎨 `Color Jitter` for simulating lighting changes  
- 📐 `Normalization` using ImageNet stats

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

During **testing**, I disabled augmentations and used only resizing and normalization to maintain consistency in evaluation.

> These augmentations helped the model learn under noisy conditions that are likely in real-world field scenarios — such as leaves photographed from different angles, under different lighting, or with camera imperfections.

### 🔍 Augmented Sample Preview

![Augmented Samples](https://github.com/user-attachments/assets/175695fe-d01c-41c5-a6d8-779c86dc35c6)

---

## 📊 Training Details

This model was trained using the following setup:

```yaml
Epochs: 10
Batch Size: 16
Optimizer: AdamW
Learning Rate: 5e-5
Loss Function: CrossEntropyLoss
```

- All training was done on a standard GPU setup.
- The final model is saved as: `swin_transformer_model.pth`

---

## ✅ Evaluation

The model was evaluated on a separate test set using:

- Accuracy  
- Precision, Recall, and F1-score (for each class)
- Confusion Matrix to understand misclassification patterns

```bash
python model.py   # Train the model
python test.py    # Evaluate on the test set
```

📉 **Confusion Matrix Visualization:**

![Confusion Matrix](https://github.com/user-attachments/assets/240c6403-a5b7-4336-8b44-9cca532cc942).

As shown above, the confusion matrix provides valuable insights into how the model handles inter-class similarity — a real challenge in leaf stress classification.

---

## 💬 Final Thoughts

This project isn't just a technical exercise — it reflects how AI can play a practical role in sustainable agriculture. By building a robust, fine-tuned model based on real data, I wanted to demonstrate how deep learning models can be deployed to support farmers in early disease detection and nutrient management.

I approached this project not just as a developer, but as someone who genuinely believes in the power of AI to solve real-world problems.

---
