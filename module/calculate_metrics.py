import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
def calculate_metrics(cm, dataset_type="Test"):

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)


    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)


    TPR = np.where(TP + FN != 0, TP / (TP + FN), 0)  # Recall or Sensitivity
    TNR = np.where(TN + FP != 0, TN / (TN + FP), 0)  # Specificity
    PPV = np.where(TP + FP != 0, TP / (TP + FP), 0)  # Precision
    NPV = np.where(TN + FN != 0, TN / (TN + FN), 0)  # Negative Predictive Value
    FPR = np.where(FP + TN != 0, FP / (FP + TN), 0)  # Fall-out or False Positive Rate
    FNR = np.where(FN + TP != 0, FN / (FN + TP), 0)  # False Negative Rate
    FDR = np.where(FP + TP != 0, FP / (FP + TP), 0)  # False Discovery Rate

    accuracy = accuracy_score(cm.sum(axis=0), cm.sum(axis=1))


    print(f"--- {dataset_type} Metrics ---")


    return {
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,

    }

