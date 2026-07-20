"""
HSAQ Training Monitor — gráficos especializados de sparsity adaptativa
Uso: python scripts/plot_hsaq.py --log outputs/training_log.csv
"""
import argparse, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(log_path):
    """Extrae métricas HSAQ del log de entrenamiento."""
    steps, losses, accs, sparsities, thresholds = [], [], [], [], []
    
    with open(log_path) as f:
        for line in f:
            m = re.search(r'E\d+/\d+ \[(\d+)/', line)
            if not m:
                continue
            step = int(m.group(1))
            
            loss_m = re.search(r'loss=([\d.]+)', line)
            acc_m = re.search(r'acc=([\d.]+)', line)
            sparse_m = re.search(r'sparse=([\d.]+)', line)
            
            if loss_m:
                steps.append(step)
                losses.append(float(loss_m.group(1)))
                accs.append(float(acc_m.group(1)) if acc_m else 0)
                sparsities.append(float(sparse_m.group(1)) if sparse_m else 0)
    
    return steps, losses, accs, sparsities


def plot_hsaq_metrics(steps, losses, accs, sparsities, output_dir):
    """Genera gráficos especializados de HSAQ."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss
    axes[0, 0].plot(steps, losses, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (con HSAQ)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy
    axes[0, 1].plot(steps, accs, 'g-', alpha=0.7)
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Token Prediction Accuracy (con HSAQ)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sparsity real vs target
    target_sparsity = 0.3  # config
    axes[1, 0].axhline(y=target_sparsity, color='r', linestyle='--', label=f'Target sparsity={target_sparsity}')
    if sparsities and any(s > 0 for s in sparsities):
        axes[1, 0].plot(steps, sparsities, 'orange', alpha=0.7, label='Real sparsity')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Sparsity Rate')
    axes[1, 0].set_title('HSAQ: Sparsity real vs target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. HSAQ Efficiency (sparsity gain vs loss)
    if sparsities and len(steps) == len(sparsities):
        efficiency = [s * 100 for s in sparsities]  # %
        axes[1, 1].plot(steps, efficiency, 'purple', alpha=0.7)
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Activaciones enmascaradas (%)')
        axes[1, 1].set_title('HSAQ: % de activaciones eliminadas por batch')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'hsaq_training_metrics.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráficos HSAQ guardados: {path}")


def plot_hsaq_heatmap(steps, sparsities, output_dir):
    """Heatmap de actividad HSAQ durante entrenamiento."""
    if len(sparsities) < 10:
        return
    
    # Crear matriz de actividad (simulada por capas)
    n_layers = 6  # embedding + 4 transformers + snn + ssm + jepa
    chunk = max(1, len(sparsities) // 100)
    sparse_avg = [np.mean(sparsities[i:i+chunk]) for i in range(0, len(sparsities), chunk)]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    layers_data = np.random.uniform(0.25, 0.35, (len(sparse_avg), n_layers))
    im = ax.imshow(layers_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Training Progress')
    ax.set_ylabel('HSAQ Layer')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(['Emb', 'T1', 'T2', 'T3', 'T4', 'SNN', 'SSM', 'JEPA'][:n_layers])
    ax.set_title('HSAQ Sparsity Heatmap por capa (simulado)')
    plt.colorbar(im, ax=ax, label='Sparsity rate')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'hsaq_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Heatmap HSAQ guardado: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-l', type=str, required=True, help='Ruta al CSV de entrenamiento')
    parser.add_argument('--output', '-o', type=str, default='docs/plots', help='Directorio de salida')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    steps, losses, accs, sparsities = parse_log(args.log)
    
    if not steps:
        print("No se encontraron datos en el log. Verifica la ruta.")
        exit(1)
    
    print(f"Datos cargados: {len(steps)} puntos")
    plot_hsaq_metrics(steps, losses, accs, sparsities, args.output)
    plot_hsaq_heatmap(steps, sparsities, args.output)
