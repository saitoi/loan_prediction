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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

df = pd.read_csv("df_final.csv")
y = df["last_loan_status"].values
X = df.drop(columns=["last_loan_status"])

with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)} (proporção: {np.bincount(y)/len(y)})")

def elapsed_time_min_sec(end, start):
    elapsed_time = end - start
    min = int(elapsed_time // 60)
    sec = elapsed_time % 60
    return min, sec

def hyperparameter_tuning_cv():
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COM CROSS-VALIDATION")
    print("="*50)

    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    config_results = defaultdict(list)
    start_bestparam_search = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )

        start_grid_search = time.perf_counter()
        grid_search.fit(X_train, y_train)
        elapsed_time_gridsearch = elapsed_time_min_sec(time.perf_counter(), start_grid_search)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        config_key = str(grid_search.best_params_)
        config_results[config_key].append({
            'fold': fold,
            'accuracy': acc,
            'f1_score': f1,
            'params': grid_search.best_params_
        })

        print(f"Fold {fold:02d}: Acc={acc:.4f}, F1={f1:.4f}, Best params: {grid_search.best_params_}")
        print(f"Tempo: {elapsed_time_gridsearch[0]}min {elapsed_time_gridsearch[1]:.2f}s")

    elapsed_time_bestparam_search = elapsed_time_min_sec(time.perf_counter(), start_bestparam_search)
    config_summary = {}
    for config, results in config_results.items():
        accs = [r['accuracy'] for r in results]
        f1s = [r['f1_score'] for r in results]
        config_summary[config] = {
            'count': len(results),
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'params': results[0]['params']
        }

    best_config = max(config_summary.items(), key=lambda x: x[1]['mean_f1'])

    print(f"\n=== MELHOR CONFIGURAÇÃO ===")
    print(f"Parâmetros: {best_config[1]['params']}")
    print(f"Apareceu em {best_config[1]['count']}/30 folds")
    print(f"Accuracy: {best_config[1]['mean_accuracy']:.4f} ± {best_config[1]['std_accuracy']:.4f}")
    print(f"F1-Score: {best_config[1]['mean_f1']:.4f} ± {best_config[1]['std_f1']:.4f}")
    print(f"Tempo Total de Busca: {elapsed_time_bestparam_search[0]}min {elapsed_time_bestparam_search[1]:.2f}s")

    return best_config[1]['params']

def evaluate_fixed_params(params):
    print(f"\n" + "="*50)
    print("AVALIAÇÃO COM PARÂMETROS OTIMIZADOS")
    print("="*50)

    results = []
    all_y_true = []
    all_y_pred = []

    start_evaluation_time = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        model = DecisionTreeClassifier(**params, random_state=42)

        start_training_fit = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed_time_fit = elapsed_time_min_sec(time.perf_counter(), start_training_fit)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({'accuracy': acc, 'f1_score': f1})
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold:02d}: Acc={acc:.4f}, F1={f1:.4f}, Time: {elapsed_time_fit[0]}min {elapsed_time_fit[1]:.2f}s")

    elapsed_evaluation_time = elapsed_time_min_sec(time.perf_counter(), start_evaluation_time)
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1_score'] for r in results]

    print(f"\n=== RESUMO 30-FOLDS ===")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Melhor Acc: {np.max(accs):.4f} | Pior Acc: {np.min(accs):.4f}")
    print(f"Tempo de Avaliação: {elapsed_evaluation_time[0]}min {elapsed_evaluation_time[1]:.2f}s")

    print(f"\n=== MÉTRICAS GLOBAIS ===")
    print(classification_report(all_y_true, all_y_pred))

    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Classe 0', 'Classe 1'],
                yticklabels=['Classe 0', 'Classe 1'])
    plt.title("Matriz de Confusão (30-fold CV)")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    start_exec = time.perf_counter()

    best_params = hyperparameter_tuning_cv()
    final_results = evaluate_fixed_params(best_params)

    print(f"\n" + "="*50)
    print("COMPARAÇÃO COM BASELINE")
    print("="*50)

    baseline_results = evaluate_fixed_params({
        'max_depth': None,
        'min_samples_split': 2,
        'criterion': 'gini'
    })

    opt_f1 = np.mean([r['f1_score'] for r in final_results])
    baseline_f1 = np.mean([r['f1_score'] for r in baseline_results])
    improvement = ((opt_f1 - baseline_f1) / baseline_f1) * 100

    print(f"\nMelhoria do F1-Score: {improvement:.2f}%")
    elapsef_exec_time = elapsed_time_min_sec(time.perf_counter(), start_exec)
    print(f"Tempo Total de Execução: {elapsef_exec_time[0]}min {elapsef_exec_time[1]:.2f}s")
