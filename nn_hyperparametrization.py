
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Carrega dados
df = pd.read_csv("df_final.csv")
y = df["last_loan_status"].values
X = df.drop(columns=["last_loan_status"])

# Carrega splits
with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)} (proporção: {np.bincount(y)/len(y)})")

# ====== ABORDAGEM 1: HYPERPARAMETER TUNING + CV ======

def hyperparameter_tuning_cv():
    """
    Faz grid search usando os splits para encontrar melhores hiperparâmetros
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COM CROSS-VALIDATION")
    print("="*50)

    # Parâmetros para grid search
    param_grid = {
        'hidden_layer_sizes': [(75, 75), (100), (200)],
        'solver': ['sgd', 'adam'],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
    }

    # Resultados por configuração
    config_results = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Grid search com validação cruzada interna (3-fold para rapidez)
        grid_search = GridSearchCV(
            MLPClassifier(max_iter=200, random_state=42),
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled, y_train)

        # Testa melhor modelo no fold atual
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Armazena resultados
        config_key = str(grid_search.best_params_)
        config_results[config_key].append({
            'fold': fold,
            'accuracy': acc,
            'f1_score': f1,
            'params': grid_search.best_params_
        })

        print(f"Fold {fold:02d}: Acc={acc:.4f}, F1={f1:.4f}, Best params: {grid_search.best_params_}")

    # Analisa qual configuração foi mais consistente
    config_summary = {}
    for config, results in config_results.items():
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_score'] for r in results]

        config_summary[config] = {
            'count': len(results),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'params': results[0]['params']
        }

    # Encontra melhor configuração
    best_config = max(config_summary.items(), key=lambda x: x[1]['mean_f1'])

    print(f"\n=== MELHOR CONFIGURAÇÃO ===")
    print(f"Parâmetros: {best_config[1]['params']}")
    print(f"Apareceu em {best_config[1]['count']}/30 folds")
    print(f"Accuracy: {best_config[1]['mean_accuracy']:.4f} ± {best_config[1]['std_accuracy']:.4f}")
    print(f"F1-Score: {best_config[1]['mean_f1']:.4f} ± {best_config[1]['std_f1']:.4f}")

    return best_config[1]['params']

# ====== ABORDAGEM 2: AVALIAÇÃO COM PARÂMETROS FIXOS ======

def evaluate_fixed_params(params):
    """
    Avalia modelo com parâmetros fixos usando todos os 30 folds
    """
    print(f"\n" + "="*50)
    print("AVALIAÇÃO COM PARÂMETROS OTIMIZADOS")
    print("="*50)

    results = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treina modelo
        model = MLPClassifier(
            **params,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({'accuracy': acc, 'f1_score': f1})
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"Fold {fold:02d}: Acc={acc:.4f}, F1={f1:.4f}")

    # Estatísticas finais
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]

    print(f"\n=== RESUMO 30-FOLDS ===")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Melhor Acc: {np.max(accuracies):.4f} | Pior Acc: {np.min(accuracies):.4f}")

    # Relatório global
    print(f"\n=== MÉTRICAS GLOBAIS ===")
    print(classification_report(all_y_true, all_y_pred))

    # Matriz de confusão
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

# ====== EXECUÇÃO ======

if __name__ == "__main__":
    # 1. Encontra melhores hiperparâmetros
    best_params = hyperparameter_tuning_cv()

    # 2. Avalia modelo otimizado
    final_results = evaluate_fixed_params(best_params)

    # 3. Comparação com modelo baseline
    print(f"\n" + "="*50)
    print("COMPARAÇÃO COM BASELINE")
    print("="*50)

    baseline_results = evaluate_fixed_params({
        'solver': 'adam',
        'hidden_layer_sizes': (75, 75),
        'activation': 'logistic'
    })

    # Análise de melhoria
    opt_f1 = np.mean([r['f1_score'] for r in final_results])
    baseline_f1 = np.mean([r['f1_score'] for r in baseline_results])

    improvement = ((opt_f1 - baseline_f1) / baseline_f1) * 100
    print(f"\nMelhoria do F1-Score: {improvement:.2f}%")
