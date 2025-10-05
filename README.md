# 🍛 Vision Transformer (ViT) – Indian Food Image Classification

This project fine-tunes a **Vision Transformer (ViT)** model (`google/vit-base-patch16-224-in21k`) to classify various **Indian food images** using the Hugging Face `transformers` and `datasets` libraries.

---

## 🧠 Project Overview

The goal of this project is to leverage transfer learning from a pre-trained Vision Transformer to recognize different Indian food dishes with high accuracy.
The dataset used is **[`rajistics/indian_food_images`](https://huggingface.co/datasets/rajistics/indian_food_images)**, available on the Hugging Face Hub.

---

## ⚙️ Tools and Frameworks Used

* 🥩 **Hugging Face Transformers** – for ViT model loading and fine-tuning
* 🧠 **PyTorch** – backend deep learning framework
* 🖾️ **Torchvision** – for image transformations and preprocessing
* 📊 **Hugging Face Datasets** – for dataset management
* 📈 **Evaluate library** – for accuracy computation
* 🚀 **Google Colab** – for GPU-based training and experimentation

---

## 🔍 Preprocessing Techniques

| Step                    | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| Image Loading           | Loaded images from Hugging Face dataset `rajistics/indian_food_images`.   |
| Resizing                | Used `RandomResizedCrop` to resize all images to 224×224 pixels.          |
| Normalization           | Applied mean and std normalization from the pretrained ViT processor.     |
| Tensor Conversion       | Converted images to PyTorch tensors using `ToTensor()`.                   |
| Transformation Pipeline | Combined steps using `Compose([RandomResizedCrop, ToTensor, Normalize])`. |

---

## 🧪 Model Architecture

* Base Model: `google/vit-base-patch16-224-in21k`
* Image size: 224 × 224
* Patch size: 16 × 16
* Classification Head: Replaced with a linear layer of size = number of classes in dataset
* Optimizer: AdamW (handled internally by Trainer)
* Evaluation Metric: Accuracy

---

## 🗮️ Training Configuration

```python
TrainingArguments(
    output_dir="train_dir",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    report_to="none"
)
```

* The model was trained on **Google Colab (GPU)**.
* W&B logging was disabled using `report_to="none"` and `!pip uninstall -y wandb`.

---

## 💾 Saving and Inference

After training, the model was saved locally:

```python
trainer.save_model("food_classification")
```

Then used for inference:

```python
from transformers import pipeline
pipe = pipeline("image-classification", model="food_classification")

from PIL import Image
import requests
from io import BytesIO

url = "https://marketplace.canva.com/fyIYY/MAGoWUfyIYY/1/tl/canva-overhead-shot-of-mushroom-pizza-MAGoWUfyIYY.jpg"
image = Image.open(BytesIO(requests.get(url).content))
pipe(image)
```

---

## 📊 Example Output

```
[{'label': 'pizza', 'score': 0.9987}]
```

---

## 🏁 Results Summary

| Metric   | Score                 |
| -------- | --------------------- |
| Accuracy | ~XX% (after 4 epochs) |

*(Replace XX% with your actual result once you evaluate.)*

---

---

## 🚀 How to Run

```bash
# Clone repo

# Install dependencies

# Run training script
```

---

## 🧑‍💻 Author

**Ravindu Shehan Induruwa**
Undergraduate at University of Ruhuna, Faculty of Science
📧 mailto:ravindushehan[.ob@gmail.com](mailto:.ob@gmail.com)
🌐 [LinkedIn](https://www.linkedin.com/in/shehan-induruwa-120abc) | [GitHub](https://github.com/SHEHAN-120)

---

## 🪪 License

This project is licensed under the **MIT License** – feel free to use or modify with attribution.

---
