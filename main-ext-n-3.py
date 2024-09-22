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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # Load data
    index, excel_feature, label = extract_excel_features('excel_data/datasets-internal-external-del-name.xlsx')
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

    # Split data into training and validation sets
    train_size = 338  # 前338行作为训练集
    x_train = combined_features_tensor[:train_size]
    y_train = label_tensor[:train_size]
    x_val = combined_features_tensor[train_size:]
    y_val = label_tensor[train_size:]

    # Shuffle the training set
    torch.manual_seed(SEED)
    permutation = torch.randperm(train_size)
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

    # Initialize model
    net = Classifier(feature_dim=combined_features.shape[1], output_size=len(set(label))).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    batch_size = 16
    model_path = './pth/best_model.pth'

    k = 2  # 选择 k 值
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    all_val_accuracies = []
    all_val_precisions = []
    all_val_recalls = []
    all_val_f1s = []
    all_roc_aucs = []
    all_roc_auc_macro = []
    all_roc_auc_micro = []
    all_val_probs = []
    all_val_labels = []
    best_val_accuracy = 0.0

    for fold, (train_index, val_index) in enumerate(kf.split(x_val)):
        print(f"Fold {fold + 1}/{k}")

        # 使用当前折的验证集
        x_fold_val, y_fold_val = x_val[val_index], y_val[val_index]

        # 训练并验证模型
        cm_val, _, val_probs, _, y_val_pred, _, train_losses, val_losses = train_test(
            x_train, y_train, x_fold_val, y_fold_val,
            x_fold_val, y_fold_val,
            net, optimizer, loss_func, batch_size, model_path
        )

        # 计算验证集指标
        accuracy = accuracy_score(y_fold_val, y_val_pred)
        precision = precision_score(y_fold_val, y_val_pred, average='weighted')
        recall = recall_score(y_fold_val, y_val_pred, average='weighted')
        f1 = f1_score(y_fold_val, y_val_pred, average='weighted')
        roc_auc = plot_roc_curve(y_fold_val, val_probs, dataset_type="Validation")

        # 保存指标
        all_val_accuracies.append(accuracy)
        all_val_precisions.append(precision)
        all_val_recalls.append(recall)
        all_val_f1s.append(f1)
        # all_roc_aucs.append(roc_auc)
        all_roc_auc_macro.append(roc_auc['macro'])
        all_roc_auc_micro.append(roc_auc['micro'])

        # 保存最优模型
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(net.state_dict(), model_path)

        # 在当前折叠后，加载最佳模型进行评估
        net.load_state_dict(torch.load(model_path))

        # 使用最佳模型进行验证集评估
        with torch.no_grad():
            val_out = net(x_fold_val.to(device))
            val_probs = F.softmax(val_out, dim=1).cpu().numpy()
            y_val_pred = torch.max(val_out, 1)[1].cpu().numpy()
            # 保存每一折的验证集标签和预测概率
            all_val_probs.append(val_probs)
            all_val_labels.append(y_fold_val)
            # 计算验证集指标
            accuracy = accuracy_score(y_fold_val, y_val_pred)
            precision = precision_score(y_fold_val, y_val_pred, average='weighted')
            recall = recall_score(y_fold_val, y_val_pred, average='weighted')
            f1 = f1_score(y_fold_val, y_val_pred, average='weighted')
            roc_auc = plot_roc_curve(y_fold_val, val_probs, dataset_type="Validation")

            # 更新保存指标
            all_val_accuracies[-1] = accuracy
            all_val_precisions[-1] = precision
            all_val_recalls[-1] = recall
            all_val_f1s[-1] = f1
            # all_roc_aucs[-1] = roc_auc
            all_roc_auc_macro[-1] = roc_auc['macro']
            all_roc_auc_micro[-1] = roc_auc['micro']
            # 保存每一折的验证集标签和预测概率
            all_val_probs.append(val_probs)
            all_val_labels.append(y_fold_val)
    # 拼接所有折的验证集标签和预测概率
    all_val_probs = np.vstack(all_val_probs)
    all_val_labels = np.hstack(all_val_labels)
    plot_final_roc_curve(all_val_labels, all_val_probs)
    # 打印平均指标
    print(f"Average Validation Metrics across {k} folds:")
    print(all_val_accuracies)
    print(
        f"Accuracy: {np.mean(all_val_accuracies):.4f} ± {np.std(all_val_accuracies):.4f}, "
        f"Precision: {np.mean(all_val_precisions):.4f} ± {np.std(all_val_precisions):.4f}, "
        f"Recall: {np.mean(all_val_recalls):.4f} ± {np.std(all_val_recalls):.4f}, "
        f"F1 Score: {np.mean(all_val_f1s):.4f} ± {np.std(all_val_f1s):.4f}, "
        # f"AUC: {np.mean(all_roc_aucs):.4f} ± {np.std(all_roc_aucs):.4f}，"
        f"AUC (macro): {np.mean(all_roc_auc_macro):.4f} ± {np.std(all_roc_auc_macro):.4f}，"
        f"AUC (micro): {np.mean(all_roc_auc_micro):.4f} ± {np.std(all_roc_auc_micro):.4f}，"
    )



if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    main()
