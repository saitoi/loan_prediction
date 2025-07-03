from scipy.stats import ttest_rel

# Tree

accuracies_tree = [
    0.9597, 0.9633, 0.9568, 0.9561, 0.9770, 0.9539, 0.9654, 0.9618, 0.9640, 0.9662,
    0.9633, 0.9647, 0.9640, 0.9618, 0.9618, 0.9618, 0.9719, 0.9654, 0.9640, 0.9518,
    0.9676, 0.9568, 0.9654, 0.9590, 0.9676, 0.9705, 0.9539, 0.9604, 0.9510, 0.9719
]

f1_scores_tree = [
    0.9598, 0.9633, 0.9568, 0.9561, 0.9769, 0.9539, 0.9654, 0.9618, 0.9641, 0.9661,
    0.9632, 0.9647, 0.9640, 0.9619, 0.9619, 0.9620, 0.9719, 0.9654, 0.9640, 0.9517,
    0.9676, 0.9568, 0.9654, 0.9590, 0.9676, 0.9705, 0.9539, 0.9605, 0.9510, 0.9719
]

# SVM

accuracies_svm = [
    0.9438, 0.9532, 0.9561, 0.9604, 0.9546, 0.9575, 0.9503, 0.9410, 0.9597, 0.9518,
    0.9525, 0.9525, 0.9532, 0.9640, 0.9518, 0.9438, 0.9604, 0.9561, 0.9597, 0.9532,
    0.9561, 0.9554, 0.9510, 0.9604, 0.9510, 0.9590, 0.9431, 0.9582, 0.9532, 0.9381
]

f1_scores_svm = [
    0.9441, 0.9533, 0.9562, 0.9605, 0.9547, 0.9576, 0.9504, 0.9413, 0.9598, 0.9519,
    0.9526, 0.9527, 0.9533, 0.9641, 0.9520, 0.9440, 0.9605, 0.9561, 0.9598, 0.9534,
    0.9562, 0.9555, 0.9511, 0.9606, 0.9512, 0.9591, 0.9433, 0.9584, 0.9534, 0.9380
]

# NN

accuracies_nn = [
    0.9518, 0.9532, 0.9604, 0.9604, 0.9510, 0.9525, 0.9590, 0.9568, 0.9546, 0.9575,
    0.9510, 0.9597, 0.9546, 0.9611, 0.9539, 0.9510, 0.9546, 0.9575, 0.9640, 0.9496,
    0.9626, 0.9590, 0.9446, 0.9575, 0.9597, 0.9633, 0.9568, 0.9626, 0.9518, 0.9626
]

f1_scores_nn = [
    0.9520, 0.9533, 0.9605, 0.9605, 0.9509, 0.9527, 0.9590, 0.9570, 0.9546, 0.9577,
    0.9512, 0.9597, 0.9547, 0.9612, 0.9540, 0.9512, 0.9546, 0.9575, 0.9641, 0.9498,
    0.9627, 0.9590, 0.9446, 0.9577, 0.9599, 0.9634, 0.9569, 0.9627, 0.9518, 0.9626
]

from scipy.stats import friedmanchisquare

stat_acc, p_acc = friedmanchisquare(accuracies_tree, accuracies_svm, accuracies_nn)

stat_f1, p_f1 = friedmanchisquare(f1_scores_tree, f1_scores_svm, f1_scores_nn)

print("Friedman test (Accuracy): chi quadrado = {:.4f}, p = {:.4f}".format(stat_acc, p_acc))
print("Friedman test (F1-score): chi quadrado = {:.4f}, p = {:.4f}".format(stat_f1, p_f1))

import numpy as np
import scikit_posthocs as sp

# organiza os dados em colunas
data_acc = np.vstack([accuracies_tree, accuracies_svm, accuracies_nn]).T
data_f1_scores = np.vstack([f1_scores_tree, f1_scores_svm, f1_scores_nn]).T

# matriz de p-values do teste de Nemenyi
p_matrix = sp.posthoc_nemenyi_friedman(data_acc)
p_matrix_a = sp.posthoc_nemenyi_friedman(data_f1_scores)

print(p_matrix)
print(p_matrix_a)

