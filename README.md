\#  LungGuard AI



\### Chest X-Ray Classification System using Deep Learning



LungGuard AI is a Deep Learning-based Computer Vision project that analyzes chest X-ray images and classifies them into:



\-  COVID-19

\-  Pneumonia

\-  Normal



\##  Features



\- Chest X-ray image classification

\- Prediction confidence

\- Class probabilities

\- Simple Streamlit web interface

\- Deep Learning CNN model



\##  Model



\- Framework: TensorFlow / Keras

\- Input Size: 224 × 224

\- Image Preprocessing \& Normalization

\- CNN-based image classification



\##  Results



| Metric | Score |

|---|---:|

| Accuracy | 93.4% |

| Precision | 94% |

| Recall | 93% |

| F1-Score | 93% |



\##  Project Structure



```text

LungGuard-AI/

│

├── Data/

│   ├── train/

│   ├── val/

│   └── test/

│

├── Notebook/

│   └── LungGuard\_AI\_Model.ipynb

│

├── app.py

├── lung\_model\_cnn.keras

├── requirements.txt

├── LungGard\_Pdf.pdf

├── .gitignore

└── README.md

&#x20;Dataset



The complete dataset is available here:



Google Drive: \[DATASET DOWNLOAD LINK]



After downloading, place the Data folder inside the project directory:



LungGuard-AI/

└── Data/

&#x20;   ├── train/

&#x20;   ├── val/

&#x20;   └── test/

&#x20;Installation

git clone https://github.com/faresalfoly68-sudo/LungGuard-AI.git

cd LungGuard-AI



python -m venv venv

venv\\Scripts\\activate



pip install -r requirements.txt

&#x20;Run the Application

streamlit run app.py



Then open the local Streamlit URL in your browser.



&#x20;Technologies



Python • TensorFlow • Keras • OpenCV • NumPy • Scikit-learn • Matplotlib • Streamlit



\## Author



Fares Waleed Alfoly

Computer Science — AI Engineer Track



