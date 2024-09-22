import pandas as pd
import numpy as np
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import numpy as np

import matplotlib.pyplot as plt



# 三分类
def plot_roc_curve(y_true, y_probs, dataset_type="Test"):
    """
    绘制ROC曲线并打印AUC值
    """
    # 定义类别名称
    class_labels = ["LN0", "LN1-3", "LN4+"]
    # class_labels = ["HER2-low", "HER2-zero", "HER2-positive"]
    # 将y_true转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    # 计算微平均 ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 打印AUC值
    print(f"{dataset_type} ROC AUC values:")
    for i in range(n_classes):
        print(f"Class {class_labels[i]} AUC: {roc_auc[i]:.2f}")
    print(f"Macro Average AUC: {roc_auc['macro']:.2f}")
    print(f"Micro Average AUC: {roc_auc['micro']:.2f}")

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    # colors = ['#6a0dad', '#b30000', '#004d4d', '#000080', '#800080']
    colors = ['#9edae5', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} ROC curve (AUC = {roc_auc[i]:.2f})')


    plt.plot(fpr["macro"], tpr["macro"], color='#000080', linestyle='-.', lw=4, label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    plt.plot(fpr["micro"], tpr["micro"], color='#800080', linestyle='--', lw=4, label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_type} ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # 返回计算的 AUC 值字典
    return roc_auc
#
# # 二分类
# def plot_roc_curve(y_true, y_probs, dataset_type="Test"):
#     """
#     绘制ROC曲线并打印AUC值
#     """
#     # 定义类别名称
#     # class_labels = ["LN-", "LN+"]
#     class_labels = ["LN1-3", "LN4+"]
#     # 假设y_probs是正类（LN+）的概率
#     if y_probs.ndim > 1:
#         y_probs = y_probs[:, 1]  # 选择正类的概率
#
#     # 计算 ROC 曲线和 AUC
#     fpr, tpr, _ = roc_curve(y_true, y_probs)
#     roc_auc = auc(fpr, tpr)
#
#     # 打印 AUC 值
#     print(f"{dataset_type} ROC AUC: {roc_auc:.2f}")
#
#     # 绘制 ROC 曲线
#     plt.figure()
#     plt.plot(fpr, tpr, color='#000080',linestyle='--', lw=4, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'{dataset_type} ROC Curve')
#     plt.legend(loc="lower right")
#     plt.show()
#
#     # 返回计算的 AUC 值
#     return roc_auc

def plot_final_roc_curve(labels, probs):
    n_classes = probs.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # class_labels = ["LN-", "LN+"]
    class_labels = ["LN0", "LN1-3", "LN4+"]
    # class_labels = ["LN1-3", "LN4+"]
    # 计算每一类的ROC曲线和AUC值
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 汇总所有类的ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["macro"], tpr["macro"], color='#AEAEAE', lw=4, label='Macro-average ROC (AUC = {:.2f})'.format(roc_auc["macro"]))

    # colors = cycle(['#9A9A9A', '#40A429', '#4D1F03', '#37BEDC'])
    # colors = cycle(['#A1C9F1', '#FFB6C1', '#FFD700', '#98FB98'])
    # colors = cycle(['#3E2723', '#1B5E20', '#0D47A1', '#4A148C'])
    # colors = cycle(['#007AFF', '#5AC8FA', '#AFEEEE', '#D1F2EB'])
    # colors = cycle(['#FF4500', '#32CD32', '#1E90FF', '#FFD700'])
    # colors = cycle(['#2F4F4F', '#708090', '#BEBEBE', '#D3D3D3'])
    # colors = cycle(['#1b9e77', '#d95f02', '#7570b3', '#e7298a'])
    # colors = cycle(['#4B0082', '#4682B4', '#87CEEB', '#2E8B57'])
    # colors = cycle(['#808080', '#B22222', '#4682B4', '#228B22'])
    # colors = cycle(['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3'])
    # colors = cycle(['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C'])
    colors = cycle(['#254689', '#B5382C', '#D18C39', '#AEAEAE'])
    # , linestyle = '--'
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4, label='{0} (AUC = {1:.2f})'.format(class_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()