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
import json
from tqdm import tqdm

def run_full_eda(data_dir='./data', num_samples=1500):
    os.makedirs('eda_results', exist_ok=True)

    # 1. ESTADÍSTICAS GLOBALES DE CANALES RGB (Todo el dataset)
    print("Calculando estadísticas globales de los canales RGB...")
    transform_rgb = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    full_dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', download=True, transform=transform_rgb)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(full_loader, desc="Procesando RGB"):
        batch_samples = images.size(0)
        images_flat = images.view(batch_samples, 3, -1)
        mean += images_flat.mean(2).sum(0)
        std += images_flat.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    stats_dict = {
        "Distribucion_Canales": {
            "Red": {"mean": float(mean[0]), "std": float(std[0]), "min": 0.0, "max": 1.0},
            "Green": {"mean": float(mean[1]), "std": float(std[1]), "min": 0.0, "max": 1.0},
            "Blue": {"mean": float(mean[2]), "std": float(std[2]), "min": 0.0, "max": 1.0}
        },
        "Dispersion_Intra_Clase": {}
    }

    # 2. ANÁLISIS TOPOLÓGICO Y S-INDEX (Muestra de 64x64)
    print("\nExtrayendo vectores para análisis topológico y dispersión...")
    transform_space = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    space_dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', transform=transform_space)

    # Nombres de clases reales de Imagenette
    class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
                   'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

    indices = np.random.choice(len(space_dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(space_dataset, indices)
    space_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    all_features = []
    all_targets = []

    for images, targets in tqdm(space_loader, desc="Extrayendo espacio"):
        features = images.view(images.size(0), -1).numpy()
        all_features.append(features)
        all_targets.append(targets.numpy())

    X_raw = np.vstack(all_features)
    y = np.hstack(all_targets)

    # Calcular S-Index del espacio crudo
    dbi_raw = davies_bouldin_score(X_raw, y)
    s_index_raw = 1.0 / dbi_raw if dbi_raw > 0 else 0.0
    print(f"\nS-Index del espacio original de píxeles: {s_index_raw:.4f}")

    # Calcular Dispersión Intra-Clase para el JSON
    for class_idx in range(10):
        class_points = X_raw[y == class_idx]
        if len(class_points) > 0:
            centroid = np.mean(class_points, axis=0)
            distances = np.linalg.norm(class_points - centroid, axis=1)
            class_name = class_names[class_idx]
            stats_dict["Dispersion_Intra_Clase"][class_name] = {
                "distancia_media_al_centroide": float(np.mean(distances)),
                "desviacion_estandar_distancia": float(np.std(distances))
            }

    # Guardar JSON
    with open('eda_results/imagenette_stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print("[!] JSON de estadísticas guardado en eda_results/imagenette_stats.json")

    # 3. GRAFICAR t-SNE
    print("\nAplicando PCA y t-SNE para visualización...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_raw)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=[class_names[label] for label in y], # Mapeamos los números a nombres en la leyenda
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
    print("[!] Gráfica guardada en eda_results/raw_space_tsne.png")
    print("EDA finalizado con éxito.")

if __name__ == '__main__':
    run_full_eda()