#  Handwritten Digit Recognition using TensorFlow & Keras

A deep learning project that recognizes handwritten digits (0–9) from the **MNIST dataset** and predicts digits from custom-drawn images using a trained neural network.

---

##  Project Overview

This project builds and trains a neural network model capable of recognizing handwritten digits. The model learns from the MNIST dataset and can be used to predict digits from any new input image provided by the user.

The model achieves **~99% accuracy** on the MNIST test dataset, with low loss and consistent convergence.

---

##  Key Features

- Trains on **MNIST dataset** (60,000 training and 10,000 testing samples).  
- Achieves high accuracy (~99%) using a simple dense neural network.  
- Visualizes **training accuracy and loss** across epochs.  
- Displays a **confusion matrix** for performance evaluation.  
- Supports **custom handwritten image prediction** via OpenCV preprocessing.  

---

##  Model Architecture

| Layer Type | Details |
|-------------|----------|
| Input Layer | Flatten (28×28 pixels) |
| Hidden Layer 1 | Dense(50), Activation: ReLU |
| Hidden Layer 2 | Dense(50), Activation: ReLU |
| Output Layer | Dense(10), Activation: Sigmoid |

> Optimizer: **Adam**  
> Loss Function: **Sparse Categorical Cross-Entropy**  
> Metric: **Accuracy**

---

##  Model Performance

### Accuracy and Loss per Epoch
![Accuracy and Loss](assets/accuracy_loss.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix indicates that most digits are correctly classified, with very few misclassifications.

---

##  Prediction Examples

| Input Image | Model Prediction |
|--------------|------------------|
| ![Digit 2](assets/predicted_2.png) | **Predicted Value: 2** |
| ![Drawn Zero](assets/drawn_zero.png) | **Predicted Value: 0** |

---

##  Technologies Used

- **Python 3.9+**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib / Seaborn**
- **OpenCV**
- **Pillow**

---

##  How It Works

1. **Data Loading & Preprocessing:**  
   Loads the MNIST dataset, scales pixel values between 0–1 for normalization.

2. **Model Training:**  
   A neural network with two hidden layers is trained using backpropagation.

3. **Evaluation:**  
   Generates accuracy, loss graphs, and confusion matrix for performance visualization.

4. **Custom Prediction:**  
   Allows you to input any hand-drawn digit image, which is preprocessed (grayscale, resized to 28×28) and classified by the trained model.

---

##  Folder Structure

```
Handwritten-Digit-Recognition/
├── assets/
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   ├── predicted_2.png
│   ├── drawn_zero.png
│   ├── raw_confusion_tensor.png
│   └── mnist_zero_example.png
├── mnist_digit_classifier.py
└── README.md
```

---

##  Results Summary

- **Final Test Accuracy:** ≈ **99.0%**
- **Training Time:** 10 epochs (~few minutes on CPU/GPU)
- **Misclassifications:** <1% (mostly between similar digits like 3/5 or 4/9)

---

##  License

This project is released under the **MIT License** — free for personal and commercial use.

---

##  Acknowledgements

- **MNIST Dataset** — Yann LeCun, Corinna Cortes, Christopher Burges  
- **TensorFlow/Keras Team** — for providing powerful deep learning APIs

---


