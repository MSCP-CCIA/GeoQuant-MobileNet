import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from tqdm import tqdm

from src.models.factory import get_arcface_model
from src.topology.metrics import calculate_s_index


def plot_arcface_latent_space(model_path, arch_name='mobilenet_v3_small', device='cuda'):
    os.makedirs('eda_results', exist_ok=True)

    # 1. Cargar el modelo base
    print(f"Cargando modelo {arch_name} desde {model_path}...")
    model, _ = get_arcface_model(arch_name, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Cargar datos de validación
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.Imagenette(root='./data', split='val', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    all_embeddings = []
    all_targets = []

    # 3. Extraer Embeddings
    print("Extrayendo representaciones del espacio latente...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            # El modelo ahora retorna directamente la salida del BatchNorm1d
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(targets.numpy())

    X = np.vstack(all_embeddings)
    y = np.hstack(all_targets)

    # Calcular métrica exacta para el título
    s_index, _, _ = calculate_s_index(torch.tensor(X), torch.tensor(y))

    # 4. Reducción con t-SNE
    print("Calculando proyección t-SNE (esto puede tomar un par de minutos)...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 5. Graficar
    print("Generando gráfico...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.8,
        s=40
    )

    plt.title(f"Espacio Latente ArcFace - {arch_name}\nS-Index: {s_index:.4f}", fontsize=16, fontweight='bold')
    plt.xlabel("Dimensión t-SNE 1", fontsize=12)
    plt.ylabel("Dimensión t-SNE 2", fontsize=12)
    plt.legend(title='Clases (Imagenette)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = f'eda_results/latent_space_{arch_name}_arcface.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[!] ¡Gráfica de representación topológica guardada en {save_path}!")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Asegúrate de que la ruta coincida con el nombre con el que se guardó tu modelo
    model_path = 'checkpoints/mobilenet_v3_small_best_arcface_fp32.pth'
    plot_arcface_latent_space(model_path, device=device)