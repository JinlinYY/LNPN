import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from component.Cli_Encoder import extract_excel_features
from component.VitP_Encoder import extract_image_features
from component.AFC_Fusion import combine_features
from metrics.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from metrics.calculate_metrics import calculate_metrics
from component.BCT_Net import Classifier
from module.addbatch import addbatch
from metrics.print_metrics import print_average_metrics, print_mean_std_metrics
from module.set_seed import set_seed
from module.train_test import train_test




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # Load data
    index, excel_feature, label = extract_excel_features('excel_data/data-LN3')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # Extract image features
    image_filenames = ['image_data/{}.bmp'.format(idx) for idx in index.astype(int)]
    image_features = extract_image_features(image_filenames)
    pca = PCA(n_components=36)
    image_features_pca = pca.fit_transform(image_features)
    image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)
    # image_features_pca_tensor = torch.tensor(image_features, dtype=torch.float32)     # no PCA

    # Combine features
    combined_features = combine_features(image_features_pca_tensor, excel_feature_tensor)
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)

    # K-fold cross-validation
    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    all_metrics = {"Validation": [], "Test": []}
    accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro = [], [], [], [], [], []
    all_y_true, all_y_probs = [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]

        net = Classifier(feature_dim=combined_features.shape[1], output_size=len(set(label))).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        batch_size = 16
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )

        # Metrics calculation
        validation_metrics = calculate_metrics(cm_val, dataset_type="Validation")
        test_metrics = calculate_metrics(cm_test, dataset_type="Test")
        all_metrics["Validation"].append(validation_metrics)
        all_metrics["Test"].append(test_metrics)

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))

        # ROC curve and AUC for the current fold
        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs)
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])
    print_mean_std_metrics(all_metrics)
    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro)
    #  ROC  AUC
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall")
    print(overall_roc_auc)

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    main()
