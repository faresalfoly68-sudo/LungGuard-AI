#  LungGuard AI

### Chest X-Ray Classification System using Deep Learning

LungGuard AI is a Deep Learning-based Computer Vision project that analyzes chest X-ray images and classifies them into COVID-19, Pneumonia, or Normal.

##  Features

- Chest X-ray classification
- Prediction confidence
- Class probabilities
- CNN-based Deep Learning model
- Streamlit web interface

##  Model

- Framework: TensorFlow / Keras
- Input Size: 224 × 224
- Image Preprocessing & Normalization
- CNN-based image classification

##  Results

| Metric | Score |
|---|---:|
| Accuracy | 93.4% |
| Precision | 94% |
| Recall | 93% |
| F1-Score | 93% |

##  Project Structure & Dataset

```text
LungGuard-AI/
│
├── Data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── Notebook/
│   └── LungGuard_AI_Model.ipynb
│
├── app.py
├── lung_model_cnn.keras
├── requirements.txt
├── LungGard_Pdf.pdf
├── .gitignore
└── README.md
```

###  Dataset

The complete dataset is available on Google Drive:

[Download Dataset](https://drive.google.com/drive/folders/1qG7yoYDHHGPVapa9jeDrB5yszLgn_Exd?usp=sharing)

After downloading, place the `Data` folder inside the project directory as shown above.

##  Installation

```bash
git clone https://github.com/faresalfoly68-sudo/LungGuard-AI.git
cd LungGuard-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

##  Run the Application

```bash
streamlit run app.py
```

Then open the Streamlit URL in your browser.

##  Technologies

Python • TensorFlow • Keras • OpenCV • NumPy • Scikit-learn • Matplotlib • Streamlit

##  Author

**Fares Waleed Alfoly**

Computer Science — AI Engineer Track
