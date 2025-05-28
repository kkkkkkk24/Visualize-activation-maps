# Visualize-activation-maps
 Visualize activation maps to understand which image regions activate CNN filters for age detection. Guidelines: You can use any of your pre trained model (made by you) for this task. GUI is not necessary for this task.
# Age Detection using CNN + Activation Map Visualization

## Overview
This project detects age from facial images using a Convolutional Neural Network trained on the UTKFace dataset. It also visualizes CNN activation maps to interpret model behavior.

## Files
- `model_training.ipynb`: Main notebook for training and evaluation
- `requirements.txt`: Dependencies
- `age_cnn_weights.pt`: Model weights (see link if large)
- `age_cnn_model.pt`: Full model (see link if large)
- `activation_maps/`: Folder with activation map outputs

## Dataset
[UTKFace Dataset](https://susanqq.github.io/UTKFace/)

## Metrics
- Accuracy (±5 years): XX%
- Mean Absolute Error: XX
- Confusion Matrix, Precision, Recall (if classification)

## Drive Links
- Model Weights: [Google Drive Link]
- Saved Model: [Google Drive Link]



#Evaluation Metrics Code
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Convert to age groups (optional)
def age_to_group(age):
    if age < 20:
        return 0
    elif age < 40:
        return 1
    elif age < 60:
        return 2
    else:
        return 3

true_groups = [age_to_group(a) for a in true_ages]
pred_groups = [age_to_group(p) for p in preds]

print("Confusion Matrix:\n", confusion_matrix(true_groups, pred_groups))
print("Precision:", precision_score(true_groups, pred_groups, average='weighted'))
print("Recall:", recall_score(true_groups, pred_groups, average='weighted'))
