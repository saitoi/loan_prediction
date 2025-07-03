import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- load and preprocess ---
df = pd.read_csv("df_final.csv")
y  = df["last_loan_status"].values
X  = df.drop(columns="last_loan_status")

# X_scaled = StandardScaler().fit_transform(X)
pca      = PCA(n_components=3).fit(X)
X_pca    = pca.transform(X)
pc1, pc2, pc3 = X_pca.T

# --- build fig with two equal subplots ---
fig = plt.figure(figsize=(12, 6), constrained_layout=True)

# 3D subplot
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.scatter(pc1[y==0], pc2[y==0], pc3[y==0], marker='s', label='Classe 0')
ax3d.scatter(pc1[y==1], pc2[y==1], pc3[y==1], marker='o', label='Classe 1')
ax3d.set_box_aspect((1, 1, 1))            # assegura aspecto 1:1:1
ax3d.set_title("PCA 3D das 3 primeiras componentes")
ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
ax3d.legend()

# 2D subplot
ax2d = fig.add_subplot(1, 2, 2)
ax2d.scatter(pc1[y==0], pc2[y==0], marker='s', label='Classe 0')
ax2d.scatter(pc1[y==1], pc2[y==1], marker='o', label='Classe 1')
ax2d.set_aspect('equal', adjustable='box')  # aspecto igual em x e y
ax2d.set_title("PCA: componentes principais (PC1 × PC2)")
ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
ax2d.legend()

plt.savefig("hiperplano_svm_unico.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

print(pca.explained_variance_ratio_)
