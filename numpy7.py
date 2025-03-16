import numpy as np

"""
Machine Learning Metrics Implementation
------------------------------------

1. Classification Metrics:
   - Accuracy: Proportion of correct predictions
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
"""

# Generate sample predictions and true values
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

# Accuracy
accuracy = np.mean(y_true == y_pred)
print("\nAccuracy:", accuracy)

"""
2. Confusion Matrix Components:
   - True Positives (TP): Correctly predicted positive cases
   - True Negatives (TN): Correctly predicted negative cases
   - False Positives (FP): Incorrectly predicted positive cases
   - False Negatives (FN): Incorrectly predicted negative cases
"""

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Add new metric calculations
def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

"""
3. ROC Curve Components:
   - True Positive Rate (Sensitivity)
   - False Positive Rate (1 - Specificity)
   - AUC (Area Under Curve): Model performance metric
"""

# ROC Curve data preparation
probas = np.random.rand(8)  # Simulated probability predictions
thresholds = np.linspace(0, 1, 20)
tpr_list = []
fpr_list = []
