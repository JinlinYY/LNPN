import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from module.extract_excel_features import extract_excel_features
from module.extract_image_features import extract_image_features
from module.combine_features import combine_features
from module.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from module.calculate_metrics import calculate_metrics
from module.MLP_Classifier import MLP_Classifier
from module.addbatch import addbatch
from module.print_metrics import print_average_metrics, print_mean_std_metrics
from module.set_seed import set_seed


def train_test(train_input, train_label, val_input, val_label, test_input, test_label, net, optimizer, loss_func, batch_size, model_path='best_model.pth'):
    traindata = addbatch(train_input, train_label, batch_size)
    best_val_accuracy = 0.0

    if torch.cuda.is_available():
        net.cuda()
        train_input, train_label = train_input.cuda(), train_label.cuda()
        val_input, val_label = val_input.cuda(), val_label.cuda()
        test_input, test_label = test_input.cuda(), test_label.cuda()

    for epoch in range(1001):
        net.train()
        for step, data in enumerate(traindata):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            out = net(inputs)
            train_loss = loss_func(out, labels)
            train_loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_out = net(val_input)
            val_loss = loss_func(val_out, val_label)
            val_prediction = torch.max(val_out, 1)[1].cpu().numpy()
            val_accuracy = accuracy_score(val_label.cpu().numpy(), val_prediction)

        print(f'Epoch [{epoch + 1}/1001], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(net.state_dict(), model_path)
            print(f'Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_accuracy:.4f}')

    net.load_state_dict(torch.load(model_path))
    print('Loaded Best Model')

    test_out = net(test_input)
    test_prediction = torch.max(test_out, 1)[1].cpu().numpy()
    test_label_np = test_label.cpu().numpy()

    cm_test = confusion_matrix(test_label_np, test_prediction)

    test_probs = F.softmax(test_out, dim=1).cpu().detach().numpy()
    y_test_pred = test_prediction.tolist()

    val_probs = F.softmax(val_out, dim=1).cpu().detach().numpy()
    y_val_pred = val_prediction.tolist()

    cm_val = confusion_matrix(val_label.cpu().numpy(), val_prediction)

    return cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20


    index, excel_feature, label = extract_excel_features('excel_data/data-LN3')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)


    image_filenames = ['image_data/{}.bmp'.format(idx) for idx in index.astype(int)]
    image_features = extract_image_features(image_filenames)
    pca = PCA(n_components=36)
    image_features_pca = pca.fit_transform(image_features)
    image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)


    combined_features = combine_features(image_features_pca_tensor, excel_feature_tensor)
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)


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

        net = MLP_Classifier(feature_dim=combined_features.shape[1], output_size=len(set(label))).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        batch_size = 16
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )


        validation_metrics = calculate_metrics(cm_val, dataset_type="Validation")
        test_metrics = calculate_metrics(cm_test, dataset_type="Test")
        all_metrics["Validation"].append(validation_metrics)
        all_metrics["Test"].append(test_metrics)

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))


        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs)
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])
    print_mean_std_metrics(all_metrics)
    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro)

    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall")
    print(overall_roc_auc)

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    main()
