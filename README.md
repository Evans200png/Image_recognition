# 🐾 Cat vs Dog Classifier
This project uses a pretrained MobileNetV2 model to classify images as either cats or dogs. The model is trained in Google Colab using the TongPython Cats vs Dogs dataset, and deployed via a simple Streamlit web app

---

## 📁 Project Structure
cat-dog-classifier/
├── see.py                  # Streamlit app
├── mobilenet_weights.h5    # Trained model weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

---

## 🚀 How to Run the App
### 1. Clone the repository or copy the files
git clone https://github.com/your-username/cat-dog-classifier.git
cd cat-dog-classifier

---

### 2. Install dependencies
pip install -r requirements.txt


### 3. Launch the Streamlit app
streamlit run see.py

---

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Top Layers:
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dropout(0.5)
- Dense(1, Sigmoid)
The model is trained in two phases:
- Freeze base and train top layers
- Unfreeze base and fine-tune with low learning rate

---

## 📦 Dataset
- Source: TongPython Cats vs Dogs
- Format: Folder of images split into cat/ and dog/
- Preprocessing: Rescaled to 224×224, normalized to [0, 1]

---

## 🖼️ Streamlit Features
- Upload JPG or PNG image
- View prediction label (Cat 🐱 or Dog 🐶)
- See confidence score
- Responsive layout with image preview

---

## ✅ Requirements
streamlit
tensorflow
numpy
Pillow

---

## 📌 Notes
- The model weights are saved using model.save_weights() to avoid serialization issues.
- Do not use model.save() for deployment unless you rebuild the exact architecture when loading.

---

### ✨ Credits
- Kaggle Dataset
- TensorFlow
- Streamlit
