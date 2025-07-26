# **Vegetable Image Classifier (VGG16)**

This project implements a vegetable image classifier using a pre-trained VGG16 model, fine-tuned for a multiclass classification task. The goal is to accurately identify different types of vegetables from images.


## **Project Overview**

This repository contains code for classifying various vegetable images. We leverage the power of transfer learning by utilizing a pre-trained VGG16 convolutional neural network and adapting its final layers to our specific dataset. The model is trained to distinguish between different vegetable categories, providing insights into its performance through classification reports and confusion matrices.

## **Setup Instructions**

To set up the project locally, follow these steps:

1. **Clone the repository (if applicable):**  
   git clone \<your-repository-url\>  
   cd vegetable-classifier

2. Install Dependencies:  
   The project relies on PyTorch, torchvision, matplotlib, seaborn, and kagglehub. You can install them using pip:  
   pip install kagglehub seaborn torchvision torch

## **Dataset**

The dataset used for this project is "Vegetable Images Multiclassification" from Kaggle.

1. Download the dataset:  
   The notebook uses kagglehub to download the dataset. The following Python code snippet can be used:  
   import kagglehub  
   path \= kagglehub.dataset\_download("airzip/vegetable-images-multiclassification")  
   print("Path to dataset files:", path)

2. Copy to a custom path:  
   The notebook then copies the dataset to a local dataset directory for easier access:  
   import shutil  
   import os  
   original\_path \= "/teamspace/studios/this\_studio/.cache/kagglehub/datasets/airzip/vegetable-images-multiclassification/versions/1"  
   custom\_path \= "dataset"  
   shutil.copytree(original\_path, custom\_path, dirs\_exist\_ok=True)  
   print("Dataset copied to:", custom\_path)

   Ensure the original\_path matches where kagglehub downloads the dataset on your system.

The dataset structure is expected to be:

dataset/  
├── Vegetable Images/  
│   ├── Vegetable Images/  
│   │   ├── train/  
│   │   │   ├── Bean/  
│   │   │   ├── Bitter\_Gourd/  
│   │   │   └── ... (other classes)  
│   │   ├── test/  
│   │   │   ├── Bean/  
│   │   │   ├── Bitter\_Gourd/  
│   │   │   └── ...  
│   │   └── validation/  
│   │       ├── Bean/  
│   │       ├── Bitter\_Gourd/  
│   │       └── ...

## **Model Architecture**

The model uses a pre-trained VGG16 network from torchvision.models. The feature extraction layers (vgg.features) are frozen to retain their learned weights, and a new custom classifier head is added for our specific classification task.

import torch.nn as nn  
from torchvision import models

\# Load pre-trained VGG16  
vgg \= models.vgg16(pretrained=True)

\# Freeze feature extraction layers  
for param in vgg.features.parameters():  
    param.requires\_grad \= False

\# Define a new classifier head  
num\_classes \= \# Number of classes in your dataset (e.g., 15 for this dataset)  
vgg.classifier \= nn.Sequential(  
    nn.Linear(25088, 512), \# Input features from VGG16's last conv layer  
    nn.ReLU(),  
    nn.Dropout(0.5),  
    nn.Linear(512, 256),  
    nn.ReLU(),  
    nn.Dropout(0.5),  
    nn.Linear(256, num\_classes)  
)

model \= vgg.to(device) \# Move model to GPU if available

## **Training and Evaluation**

The model is trained using CrossEntropyLoss as the criterion and Adam optimizer for the classifier's parameters.

* **Epochs:** 6  
* **Batch Size:** 64 for training, 32 for validation/testing.  
* **Learning Rate:** 0.001  
* **Weight Decay:** 1e-5

The training process involves iterating through the training data, performing forward and backward passes, and updating model weights. After training, the model's performance is evaluated on separate test and validation datasets using a classification report and confusion matrices.

## **Results**

### **Training Confusion Matrix**

|  | Bean | Bitter\_Gourd | Bottle\_Gourd | Brinjal | Broccoli | Cabbage | Capsicum | Carrot | Cauliflower | Cucumber | Papaya | Potato | Pumpkin | Radish | Tomato |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Bean** | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bitter\_Gourd** | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bottle\_Gourd** | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Brinjal** | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Broccoli** | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cabbage** | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Capsicum** | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Carrot** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cauliflower** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cucumber** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| **Papaya** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 |
| **Potato** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 |
| **Pumpkin** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 |
| **Radish** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **Tomato** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 |

*Note: This is a placeholder table. Please replace with your actual training confusion matrix data.*

### **Validation Confusion Matrix**

|  | Bean | Bitter\_Gourd | Bottle\_Gourd | Brinjal | Broccoli | Cabbage | Capsicum | Carrot | Cauliflower | Cucumber | Papaya | Potato | Pumpkin | Radish | Tomato |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Bean** | 75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bitter\_Gourd** | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Bottle\_Gourd** | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Brinjal** | 0 | 0 | 0 | 97 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Broccoli** | 0 | 0 | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cabbage** | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Capsicum** | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Carrot** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cauliflower** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cucumber** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 | 0 | 0 | 0 | 0 |
| **Papaya** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 95 | 0 | 0 | 0 | 0 |
| **Potato** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 96 | 0 | 0 | 0 |
| **Pumpkin** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 | 0 |
| **Radish** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| **Tomato** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

*Note: This is a placeholder table. Please replace with your actual validation confusion matrix data.*

### **Sample Predictions**

Here are some sample predictions from the model on validation images. The title indicates "T: True Class" and "P: Predicted Class". A green border/title indicates a correct prediction, while a red one indicates an incorrect prediction.
<img width="1990" height="246" alt="image" src="https://github.com/user-attachments/assets/2f7019a7-94c0-4782-a500-feab4811d992" />



*Note: This is a placeholder image. Please replace with your actual generated image display.*

## **How to Run**

1. **Ensure all dependencies are installed** as listed in the [Setup Instructions](https://www.google.com/search?q=%23setup-instructions).  
2. **Download and set up the dataset** as described in the [Dataset](https://www.google.com/search?q=%23dataset) section.  
3. **Open the Jupyter Notebook** Vegetableclassifier-vgg16.ipynb in a Jupyter environment (e.g., JupyterLab, VS Code with Jupyter extension).  
4. **Run all cells sequentially** from top to bottom. The notebook will handle data loading, model definition, training, evaluation, and visualization.
