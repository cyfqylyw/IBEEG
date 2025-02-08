import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pywt
import math
import random
import scipy.io
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve


def simple_test(model, test_loader, args):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for eeg, label in test_loader:
            eeg_ft = torch.abs(torch.fft.fft(eeg, dim=1)).float().to(args.device)
            eeg_wt = torch.Tensor([np.concatenate(pywt.wavedec(subeeg.numpy(), 'db1'), axis=1) for subeeg in eeg]).float().to(args.device)

            if args.dataset in ["isruc", "sleepedf", "hmc", "tuab", "tuev"]:
                eeg_wt = eeg_wt[:, :, :args.chunk_second * args.freq_rate].to(args.device)
                
            # eeg_wt = eeg_wt[:eeg_ft.shape[0], :eeg_ft.shape[1], :eeg_ft.shape[2]]
            eeg = eeg.float().to(args.device)
            label = label.to(args.device)

            if args.dataset in ['hinss', 'isruc']:
                eeg = (eeg - eeg.mean(dim=2, keepdim=True)) / eeg.std(dim=2, keepdim=True)
                eeg_ft = (eeg_ft - eeg_ft.mean(dim=2, keepdim=True)) / eeg_ft.std(dim=2, keepdim=True)
                eeg_wt = (eeg_wt - eeg_wt.mean(dim=2, keepdim=True)) / eeg_wt.std(dim=2, keepdim=True)

            if args.model_name in ['EEG_CNN_Network', 'EEG_Transformer_Network']:
                outputs = model(eeg)
            elif args.model_name == 'EEG_Transformer_VIB_Network':
                (mu, std), outputs = model(eeg)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network':
                (mu, std), outputs, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network_wo_FTV':
                (mu, std), outputs, eeg_projection, eeg_projection_wt = model(eeg, eeg_wt)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network_wo_WTV':
                (mu, std), outputs, eeg_projection, eeg_projection_ft = model(eeg, eeg_ft)
            elif args.model_name == 'EEG_Transformer_CL_Network':
                outputs, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(label.cpu().numpy())  # 真实标签
            y_pred.extend(predicted.cpu().numpy())  # 预测标签
            y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # 所有类别的概率

    # 计算并输出各项指标
    accuracy = (np.array(y_pred) == np.array(y_true)).mean()
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(args.num_class):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # PR AUC计算
    precision = {}
    recall = {}
    pr_auc = {}
    for i in range(args.num_class):
        precision[i], recall[i], _ = precision_recall_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
        pr_auc[i] = auc(recall[i], precision[i])  # 注意参数顺序：先recall后precision
    
    class_weights = [np.sum(np.array(y_true) == i) for i in range(args.num_class)]
    weighted_auc = np.average(list(roc_auc.values()), weights=class_weights)
    weighted_aupr = np.average(list(pr_auc.values()), weights=class_weights)  # 新增加权PR AUC

    weighted_auc2 = np.average(list(roc_auc.values()), weights=[np.sum(np.array(y_true) == i) for i in range(args.num_class)])
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Test ACC: {accuracy:.4f}, F1: {weighted_f1:.4f}, AUROC: ({weighted_auc:.4f}, {weighted_auc2:.4f}), AUPRC: {weighted_aupr:.4f}")


def test(args, model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    torch.save(model.state_dict(), f'./results/ckpt/{args.dataset}_{args.model_name}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}.pth')

    with torch.no_grad():
        for eeg, label in test_loader:
            eeg_ft = torch.abs(torch.fft.fft(eeg, dim=1)).float().to(args.device)
            eeg_wt = torch.Tensor([np.concatenate(pywt.wavedec(subeeg.numpy(), 'db1'), axis=1) for subeeg in eeg]).float().to(args.device)
            
            if args.dataset in ["isruc", "sleepedf", "hmc", "tuab", "tuev"]:
                eeg_wt = eeg_wt[:,:,:args.chunk_second * args.freq_rate].to(args.device)

            eeg = eeg.float().to(args.device)
            label = label.to(args.device)

            if args.dataset in ['hinss', 'isruc']:
                eeg = (eeg - eeg.mean(dim=2, keepdim=True)) / eeg.std(dim=2, keepdim=True)
                eeg_ft = (eeg_ft - eeg_ft.mean(dim=2, keepdim=True)) / eeg_ft.std(dim=2, keepdim=True)
                eeg_wt = (eeg_wt - eeg_wt.mean(dim=2, keepdim=True)) / eeg_wt.std(dim=2, keepdim=True)

            if args.model_name in ['EEG_CNN_Network', 'EEG_Transformer_Network']:
                outputs = model(eeg)
            elif args.model_name == 'EEG_Transformer_VIB_Network':
                (mu, std), outputs = model(eeg)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network':
                (mu, std), outputs, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network_wo_FTV':
                (mu, std), outputs, eeg_projection, eeg_projection_wt = model(eeg, eeg_wt)
            elif args.model_name == 'EEG_Transformer_CL_VIB_Network_wo_WTV':
                (mu, std), outputs, eeg_projection, eeg_projection_ft = model(eeg, eeg_ft)
            elif args.model_name == 'EEG_Transformer_CL_Network':
                outputs, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)
                
            _, predicted = torch.max(outputs, 1)

            y_true.extend(label.cpu().numpy())  # 真实标签
            y_pred.extend(predicted.cpu().numpy())  # 预测标签
            y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # 所有类别的概率

    # 计算并输出各项指标
    print('*' * 20)
    accuracy = (np.array(y_pred) == np.array(y_true)).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    np.save(f"./results/plot_data/{args.dataset}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}_y_true.npy", np.array(y_true))
    np.save(f"./results/plot_data/{args.dataset}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}_y_pred.npy", np.array(y_pred))
    np.save(f"./results/plot_data/{args.dataset}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}_y_prob.npy", np.array(y_prob))

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # 计算 ROC 曲线和 AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(10, 8))
    
    # 计算每个类别的 ROC 曲线和 AUC
    for i in range(args.num_class):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # 绘制对角线（随机分类的 AUC = 0.5）
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Multi-Class (AUCs)')
    plt.legend(loc='lower right')

    # plt.show()
    plt.savefig(f"results/ROC_plots/{args.dataset}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}.png", dpi=300)


    plt.figure(figsize=(10, 8))
    # PR AUC计算
    precision = {}
    recall = {}
    pr_auc = {}
    for i in range(args.num_class):
        precision[i], recall[i], _ = precision_recall_curve(np.array(y_true) == i, np.array(y_prob)[:, i])
        pr_auc[i] = auc(recall[i], precision[i])  # 注意参数顺序：先recall后precision
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AUPRC = {pr_auc[i]:.2f})')
    
    class_weights = [np.sum(np.array(y_true) == i) for i in range(args.num_class)]
    weighted_aupr = np.average(list(pr_auc.values()), weights=class_weights)  # 新增加权PR AUC

    mean_precision = np.unique(np.concatenate([recall[i] for i in range(args.num_class)]))
    mean_precision = np.interp(mean_precision, 
                            np.concatenate([recall[i] for i in range(args.num_class)]),
                            np.concatenate([precision[i] for i in range(args.num_class)]))
    plt.plot(mean_precision, mean_precision, 
            color='black', linestyle='--', 
            label=f'Weighted Avg (AUPRC = {weighted_aupr:.2f})')

    # 设置图表参数
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRC)')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.savefig(f"results/ROC_PR/{args.dataset}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}.png", dpi=300)
    

    # 计算 macro-average 和 weighted-average AUC
    macro_auc = np.mean(list(roc_auc.values()))
    weighted_auc = np.average(list(roc_auc.values()), weights=[np.sum(np.array(y_true) == i) for i in range(args.num_class)])

    print(f"Macro Average AUC: {macro_auc:.4f}")
    print(f"Weighted Average AUC: {weighted_auc:.4f}")
    
    # 计算 Precision, Recall, F1 Score
    precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 输出分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(args.num_class)]))

    # 计算平衡准确率
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 计算宏平均F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\nMacro F1 Score: {macro_f1:.4f}")

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"Micro F1 Score: {micro_f1:.4f}")

    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
