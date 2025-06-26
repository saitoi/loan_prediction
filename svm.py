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
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("df_final_1.csv")

y = df["last_loan_status"].values
X = df.drop(columns=["last_loan_status"])

with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

results = []

for fold, (train_idx, test_idx) in enumerate(splits, start=1):
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_test  = X.iloc[test_idx]
    y_test  = y[test_idx]

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Fold {fold:02d} ---  Acurácia: {acc:.4f}")
    results.append(acc)

mean_acc = np.mean(results)
std_acc  = np.std(results)
print(f"\n=== RESUMO 30-FOLDS ===")
print(f"Acurácia média: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Melhor: {np.max(results):.4f}  |  Pior: {np.min(results):.4f}")

all_y_true = []
all_y_pred = []

for train_idx, test_idx in splits:
    model = SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=42)
    model.fit(X.iloc[train_idx], y[train_idx])
    preds = model.predict(X.iloc[test_idx])
    all_y_true.extend(y[test_idx])
    all_y_pred.extend(preds)

print("\n=== MÉTRICAS GLOBAIS ===")
print(classification_report(all_y_true, all_y_pred))

cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(y))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title("Confusion Matrix (global 30-fold CV)")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()
