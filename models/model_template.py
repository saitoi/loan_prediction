# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("df_final.csv")
y  = df["last_loan_status"].values
X  = df.drop(columns=["last_loan_status"])

with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

# defina seus modelos aqui
MODELS = {
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=42),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
    # adicione outros modelos se desejar
}

for name, model in MODELS.items():
    print(f"\n=== MODELO: {name} ===")
    fold_acc = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx],    y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"Fold {fold:02d} — Acurácia: {acc:.4f}")
        fold_acc.append(acc)
        all_true.extend(y_test)
        all_pred.extend(preds)

    mean_acc, std_acc = np.mean(fold_acc), np.std(fold_acc)
    print(f"Acurácia média: {mean_acc:.4f} ± {std_acc:.4f}")
    print("\nMétricas globais:")
    print(classification_report(all_true, all_pred))

    cm = confusion_matrix(all_true, all_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.show()
