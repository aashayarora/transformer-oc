import json
import random
import os

import numpy as np
import torch

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import mplhep as hep
hep.style.use(hep.style.CMS)
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = next(model.parameters()).device

        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.train_loader):
                node_features, edge_index, edge_features, y = data.x, data.edge_index, data.edge_attr, data.y
                node_features, edge_index, edge_features, y = [i.to(self.device) for i in (node_features, edge_index, edge_features, y)]
                pred = self.model(node_features, edge_features, edge_index).squeeze()
                train_predictions.append(pred)
                train_labels.append(y)
                if i >= 50:
                    break

            for i, data in enumerate(self.test_loader):
                node_features, edge_index, edge_features, y = data.x, data.edge_index, data.edge_attr, data.y
                node_features, edge_index, edge_features, y = [i.to(self.device) for i in (node_features, edge_index, edge_features, y)]
                pred = self.model(node_features, edge_features, edge_index).squeeze()
                test_predictions.append(pred)
                test_labels.append(y)
                if i >= 50:
                    break

        self.train_predictions = torch.cat(train_predictions).cpu().numpy()
        self.train_labels = torch.cat(train_labels).cpu().numpy()
        self.test_predictions = torch.cat(test_predictions).cpu().numpy()
        self.test_labels = torch.cat(test_labels).cpu().numpy()

    def plot_precision_recall_curve(self, save_path='precision_recall_curve.png'):
        self.train_precision, self.train_recall, _ = precision_recall_curve(self.train_labels, self.train_predictions)
        self.test_precision, self.test_recall, _ = precision_recall_curve(self.test_labels, self.test_predictions)

        fig, ax = plt.subplots()
        ax.plot(self.train_recall[5:], self.train_precision[5:])
        ax.plot(self.test_recall[5:], self.test_precision[5:])
        ax.set_xlabel('Recall (Efficiency)')
        ax.set_ylabel('Precision (1 - Fake Rate)')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')

        plt.savefig(save_path)

    def plot_roc_curve(self, save_path='roc_curve.png'):
        self.train_fpr, self.train_tpr, _ = roc_curve(self.train_labels, self.train_predictions)
        self.train_auc = roc_auc_score(self.train_labels, self.train_predictions)
        self.test_fpr, self.test_tpr, _ = roc_curve(self.test_labels, self.test_predictions)
        self.test_auc = roc_auc_score(self.test_labels, self.test_predictions)

        fig, ax = plt.subplots()
        ax.plot(self.train_fpr, self.train_tpr, label=f'Train AUC = {self.train_auc:.2f}')
        ax.plot(self.test_fpr, self.test_tpr, label=f'Test AUC = {self.test_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='upper right')

        plt.savefig(save_path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False