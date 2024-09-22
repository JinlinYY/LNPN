from sklearn.metrics import roc_curve, auc
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, MiniBatchSparsePCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from component.Cli_Encoder import extract_excel_features
from component.VitP_Encoder import extract_image_features
from component.AFC_Fusion import combine_features
from metrics.plot_roc_curve import plot_roc_curve, plot_final_roc_curve
from module.inputtotensor import inputtotensor
from component.BCT_Net import Classifier
from module.addbatch import addbatch
from module.set_seed import set_seed
from module.train_test import train_test, plot_epoch_losses
from module.my_loss import FocalLoss, AdaptiveWeightedFocalLoss
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # Load data
    index, excel_feature, label = extract_excel_features('excel_data/datasets-internal-del-name-34.xlsx')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # Extract image features
    image_filenames = ['image_data_internal_external/{}.bmp'.format(idx) for idx in index.astype(int)]
    image_features = extract_image_features(image_filenames)
    pca = MiniBatchSparsePCA(n_components=10)
    image_features_pca = pca.fit_transform(image_features)
    image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)

    excel_feature_pca_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # Combine features
    combined_features = combine_features(image_features_pca_tensor, excel_feature_pca_tensor)
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)

    k = 8  # Number of folds for cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    all_val_accuracies = []
    all_val_precisions = []
    all_val_recalls = []
    all_val_f1s = []
    all_roc_aucs = []
    all_val_probs = []
    all_val_labels = []
    best_val_accuracy = 0.0
    model_path = './pth/best_model.pth'

    for fold, (train_index, val_index) in enumerate(kf.split(combined_features_tensor)):
        print(f"Fold {fold + 1}/{k}")

        x_train, x_val = combined_features_tensor[train_index], combined_features_tensor[val_index]
        y_train, y_val = label_tensor[train_index], label_tensor[val_index]

        # Initialize model
        net = Classifier(feature_dim=combined_features_tensor.shape[1], output_size=len(set(label))).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss().to(device)

        batch_size = 16

        # Train and validate the model
        cm_val, _, val_probs, _, y_val_pred, _, train_losses, val_losses = train_test(
            x_train, y_train, x_val, y_val,
            x_val, y_val,
            net, optimizer, loss_func, batch_size, model_path
        )

        # Calculate validation metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='weighted')
        recall = recall_score(y_val, y_val_pred, average='weighted')
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        roc_auc = plot_roc_curve(y_val, val_probs, dataset_type="Validation")

        # Save metrics
        all_val_accuracies.append(accuracy)
        all_val_precisions.append(precision)
        all_val_recalls.append(recall)
        all_val_f1s.append(f1)
        all_roc_aucs.append(roc_auc)

        # Save each fold's validation labels and probabilities
        all_val_probs.append(val_probs)
        all_val_labels.append(y_val)

        # Save the best model
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(net.state_dict(), model_path)

    # Concatenate all folds' validation labels and probabilities
    all_val_probs = np.vstack(all_val_probs)
    all_val_labels = np.hstack(all_val_labels)

    # Plot final ROC curve
    plot_final_roc_curve(all_val_labels, all_val_probs)

    # Print average metrics
    print(f"Average Validation Metrics across {k} folds:")
    print(
        f"Accuracy: {np.mean(all_val_accuracies):.4f} ± {np.std(all_val_accuracies):.4f}, "
        f"Precision: {np.mean(all_val_precisions):.4f} ± {np.std(all_val_precisions):.4f}, "
        f"Recall: {np.mean(all_val_recalls):.4f} ± {np.std(all_val_recalls):.4f}, "
        f"F1 Score: {np.mean(all_val_f1s):.4f} ± {np.std(all_val_f1s):.4f}, "
        f"AUC: {np.mean(all_roc_aucs):.4f} ± {np.std(all_roc_aucs):.4f},"
    )

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    main()
