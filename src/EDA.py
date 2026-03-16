import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import os
from tqdm import tqdm


def run_space_eda(data_dir='./data', num_samples=1500):
    os.makedirs('eda_results', exist_ok=True)

    # Redimensionamos a 64x64 solo para que las matemáticas del EDA no colapsen la RAM
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    print("Cargando Imagenette para Análisis del Espacio Original...")
    dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', download=True, transform=transform)

    # Extraemos una muestra aleatoria para el análisis topológico
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    all_features = []
    all_targets = []

    print("Extrayendo vectores del espacio de píxeles...")
    for images, targets in tqdm(loader):
        # Aplanamos la imagen: 64 * 64 * 3 = 12,288 dimensiones
        features = images.view(images.size(0), -1).numpy()
        all_features.append(features)
        all_targets.append(targets.numpy())

    X_raw = np.vstack(all_features)
    y = np.hstack(all_targets)

    # 1. Calcular el S-Index del espacio crudo
    print("\nCalculando métricas topológicas base...")
    dbi_raw = davies_bouldin_score(X_raw, y)
    s_index_raw = 1.0 / dbi_raw if dbi_raw > 0 else 0.0
    print(f"S-Index del espacio original de píxeles: {s_index_raw:.4f}")

    # 2. Reducción de Dimensionalidad para Visualización
    print("Aplicando PCA para reducir ruido inicial...")
    # PCA rápido para bajar a 50 dimensiones antes del t-SNE (Mejora la calidad visual y la velocidad)
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_raw)

    print("Calculando t-SNE para proyección 2D...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # 3. Graficar el Caos
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.7,
        s=30
    )
    plt.title(f"t-SNE del Espacio Original de Píxeles (S-Index: {s_index_raw:.4f})", fontsize=14)
    plt.xlabel("Dimensión t-SNE 1")
    plt.ylabel("Dimensión t-SNE 2")
    plt.legend(title='Clases (Imagenette)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig('eda_results/raw_space_tsne.png', dpi=300)
    print("\n[!] Gráfica guardada en eda_results/raw_space_tsne.png")
    print("EDA de espacio finalizado.")


if __name__ == '__main__':
    run_space_eda()