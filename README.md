
# Mango Leaf Disease Classification (Transfer Learning)

A deep learning project that classifies **leaf images as Healthy or Diseased** using **Transfer Learning (MobileNetV2)**.
The project includes dataset preparation, model training, and a **Streamlit web app** for real-time prediction.


## Project Overview

This project uses a **pre-trained MobileNetV2 CNN model** fine-tuned on a mango leaf dataset to detect whether a leaf is **healthy or diseased**.
A user-friendly Streamlit application allows users to upload leaf images and get instant predictions with confidence scores.


## Model Details

 **Architecture**: MobileNetV2 (Transfer Learning)
 **Input Size**: 128 × 128 RGB images
 **Output**: Binary classification (Healthy / Diseased)
 **Loss Function**: Binary Crossentropy
 **Optimizer**: Adam
 **Evaluation Metric**: Accuracy

## 📂 Project Structure

```
leaf_cnn_project/
│
├── app.py                  # Streamlit application
├── cnn_model.py            # Model training using transfer learning
├── split_dataset.py        # Dataset splitting script
├── requirements.txt
├── README.md
│
├── models/
│   └── leaf_model_transfer.h5
│
├── data/                   # (Not included in repo)
│   ├── train/
│   └── test/
│
├── all_images/             # Original dataset (not included)
```

> ⚠️ **Dataset folders are excluded from GitHub using `.gitignore`.**


##  Dataset Preparation

The dataset is expected to be organized as:

all_images/
├── diseased/
├── healthy/


Run the dataset split script:

python split_dataset.py


This will create:

data/
├── train/
└── test/


##  Model Training

To train the model using transfer learning:

python cnn_model.py


After training, the model is saved as:

models/leaf_model_transfer.h5


## Run the Streamlit App

To launch the web application:

streamlit run app.py


Upload a leaf image and the app will display:

* Predicted class (Healthy / Diseased)
* Confidence score
* Raw sigmoid probability


## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* PIL
* Streamlit
* Matplotlib


##  Notes

* Large image datasets are not uploaded to GitHub.
* Trained model files are included for easy testing.
* Class labels are inferred directly from training folder names.


##  License

This project is open-source and intended for **educational and learning purposes**.


