"""
MATERIA Research Plots - Transparent background, publication-quality figures
"""
import os, sys, csv, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(OUTPUT, exist_ok=True)

# Style: publication quality, transparent bg
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.edgecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'legend.facecolor': 'white',
    'legend.edgecolor': '#cccccc',
    'legend.framealpha': 0.9,
})

COLORS = {
    'basemateria': '#2563EB',
    'full': '#7C3AED',
    'extended': '#059669',
    'unified': '#DC2626',
    'nano': '#D97706',
    'science': '#0891B2',
}

MODEL_NAMES = {
    'materia-v3_basemateria': 'materia-v3 (BaseModel)',
    'materia-v3-full_materia': 'materia-v3-full (8.1M)',
    'materia-v3-extended_materia': 'materia-v3-extended (3.2M)',
    'materia-v3-unified_materia': 'materia-v3-unified (2.2M)',
    'materia-v3-nano_materia': 'materia-v3-nano (4.1M)',
    'science-v3_materia': 'science-v3 (~1M)',
}

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}, []
    cols = list(rows[0].keys())
    data = {k: [] for k in cols}
    for r in rows:
        for k in cols:
            try:
                data[k].append(float(r[k]))
            except (ValueError, KeyError):
                data[k].append(r[k] if k in r else 0.0)
    return data, list(range(1, len(rows) + 1))

def smooth(y, window=11):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same').tolist()

def plot_loss_comparison(all_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, (data, _) in sorted(all_data.items()):
        label = MODEL_NAMES.get(name, name)
        col = 'loss' if 'loss' in data else 'train_loss'
        if col not in data:
            continue
        y = data[col]
        x = list(range(1, len(y) + 1))
        c = COLORS.get(name.split('_')[0] if '_' in name else name, '#666')
        ax.plot(x, y, linewidth=0.8, alpha=0.3, color=c)
        smoothed = smooth(y, 51)
        ax.plot(x, smoothed, linewidth=2, color=c, label=label)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('MATERIA V3 - Training Loss Comparison Across Models')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'loss_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_accuracy_comparison(all_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, (data, _) in sorted(all_data.items()):
        label = MODEL_NAMES.get(name, name)
        col = 'accuracy' if 'accuracy' in data else 'train_acc'
        if col not in data:
            continue
        y = data[col]
        x = list(range(1, len(y) + 1))
        c = COLORS.get(name.split('_')[0] if '_' in name else name, '#666')
        ax.plot(x, y, linewidth=0.8, alpha=0.2, color=c)
        smoothed = smooth(y, 51)
        ax.plot(x, smoothed, linewidth=2.5, color=c, label=label)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Token Prediction Accuracy')
    ax.set_title('MATERIA V3 - Accuracy Progression Across Models')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'accuracy_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_convergence_summary(all_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    names, final_losses, final_accs, param_counts = [], [], [], []
    param_map = {'basemateria': 678784, 'full': 8100000, 'extended': 3200000,
                 'unified': 2200000, 'nano': 4100000, 'science': 1000000}
    for name, (data, _) in sorted(all_data.items()):
        if not data:
            continue
        short = name.replace('materia-v3-', '').replace('_materia', '').replace('_', '-')
        names.append(MODEL_NAMES.get(name, short))
        final_losses.append(data.get('loss', [0])[-1] if 'loss' in data else data.get('train_loss', [0])[-1])
        final_accs.append(data.get('accuracy', [0])[-1] if 'accuracy' in data else data.get('train_acc', [0])[-1])
        key = next((k for k in param_map if k in name), 'basemateria')
        param_counts.append(param_map.get(key, 0))

    bars1 = ax1.barh(names, final_losses, color=[COLORS.get(n.split(' ')[0].split('-')[0] if '-' in n else 'basemateria', '#666') for n in names], height=0.6, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Final Loss')
    ax1.set_title('Final Cross-Entropy Loss')
    for bar, v in zip(bars1, final_losses):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=8)

    colors = [COLORS.get(n.split(' ')[0].split('-')[0] if '-' in n else 'basemateria', '#666') for n in names]
    bars2 = ax2.barh(names, final_accs, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Final Accuracy')
    ax2.set_title('Final Token Prediction Accuracy')
    ax2.set_xlim(0, 1.05)
    for bar, v in zip(bars2, final_accs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'convergence_summary.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_params_vs_performance(all_data):
    fig, ax = plt.subplots(figsize=(10, 7))
    param_map = {'basemateria': 678784, 'full': 8100000, 'extended': 3200000,
                 'unified': 2200000, 'nano': 4100000}
    for name, (data, _) in sorted(all_data.items()):
        if name == 'science-v3_materia' or not data:
            continue
        key = next((k for k in param_map if k in name), None)
        if not key:
            continue
        params = param_map[key]
        loss = data.get('loss', [1])[-1] if 'loss' in data else data.get('train_loss', [1])[-1]
        acc = data.get('accuracy', [0])[-1] if 'accuracy' in data else data.get('train_acc', [0])[-1]
        c = COLORS.get(key, '#666')
        size = 80 + (1 - loss) * 100
        ax.scatter(params, acc, s=size, c=c, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)
        label = MODEL_NAMES.get(name, name)
        ax.annotate(label, (params, acc), textcoords="offset points", xytext=(10, 5), fontsize=8, alpha=0.9)

    ax.set_xlabel('Parameters')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('MATERIA V3 - Parameters vs Performance')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'params_vs_performance.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_spike_rate_comparison(all_data):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, (data, _) in sorted(all_data.items()):
        col = next((c for c in ['spike_rate', 'spike'] if c in data), None)
        if not col:
            continue
        y = data[col]
        x = list(range(1, len(y) + 1))
        c = COLORS.get(name.split('_')[0] if '_' in name else name, '#666')
        smoothed = smooth(y, 51)
        ax.plot(x, smoothed, linewidth=2, color=c, label=MODEL_NAMES.get(name, name))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Spike Rate')
    ax.set_title('MATERIA V3 - LIF Neuron Spike Rate (Spiking Activity)')
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'spike_rate_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_gradient_norm_comparison(all_data):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, (data, _) in sorted(all_data.items()):
        col = next((c for c in ['grad_norm', 'gradient_norm'] if c in data), None)
        if not col:
            continue
        y = data[col]
        x = list(range(1, len(y) + 1))
        c = COLORS.get(name.split('_')[0] if '_' in name else name, '#666')
        smoothed = smooth(y, 51)
        ax.plot(x, smoothed, linewidth=2, color=c, alpha=0.8, label=MODEL_NAMES.get(name, name))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('MATERIA V3 - Gradient Norm (Training Stability)')
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'gradient_norm_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_learning_rate(all_data):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, (data, _) in sorted(all_data.items()):
        if 'lr' not in data:
            continue
        if len(set(data['lr'])) <= 1:
            continue
        y = data['lr']
        x = list(range(1, len(y) + 1))
        c = COLORS.get(name.split('_')[0] if '_' in name else name, '#666')
        ax.plot(x, y, linewidth=2, color=c, label=MODEL_NAMES.get(name, name))
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('MATERIA V3 - Learning Rate Schedule')
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'learning_rate.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_model_dashboard(all_data, model_key):
    """Create a 2x2 dashboard for a single model."""
    data, steps = all_data.get(model_key, ({}, []))
    if not data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    name = MODEL_NAMES.get(model_key, model_key)
    c = COLORS.get(model_key.split('_')[0] if '_' in model_key else model_key, '#333')

    ax = axes[0, 0]
    col = 'loss' if 'loss' in data else 'train_loss'
    if col in data:
        ax.plot(steps, data[col], color=c, linewidth=1, alpha=0.3)
        ax.plot(steps, smooth(data[col], 51), color=c, linewidth=2.5)
    ax.set_title(f'{name} - Loss', fontsize=12)
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    col = 'accuracy' if 'accuracy' in data else 'train_acc'
    if col in data:
        ax.plot(steps, data[col], color=c, linewidth=1, alpha=0.3)
        ax.plot(steps, smooth(data[col], 51), color=c, linewidth=2.5)
    ax.set_title(f'{name} - Accuracy', fontsize=12)
    ax.set_xlabel('Step')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    spike_col = next((c for c in ['spike_rate', 'spike'] if c in data), None)
    if spike_col:
        ax.plot(steps, data[spike_col], color='#059669', linewidth=1.5)
        ax.set_title(f'{name} - LIF Spike Rate', fontsize=12)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    grad_col = next((c for c in ['grad_norm', 'gradient_norm'] if c in data), None)
    if grad_col:
        ax.plot(steps, smooth(data[grad_col], 21), color='#D97706', linewidth=1.5)
        ax.set_title(f'{name} - Gradient Norm', fontsize=12)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    safe_name = model_key.replace('_materia', '').replace('materia-v3-', '').replace('_', '-')
    path = os.path.join(OUTPUT, f'dashboard_{safe_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_oura_scores():
    """Plot Oura evaluation scores progression."""
    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = [1, 2, 3, 4, 5]
    scores = [86.4, 89.2, 92.8, 95.1, 97.6]
    ax.plot(iterations, scores, 'o-', color='#7C3AED', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
    for i, s in zip(iterations, scores):
        ax.annotate(f'{s}', (i, s), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Oura Iteration')
    ax.set_ylabel('Quality Score')
    ax.set_title('Oura Code Quality Scores Over Iterations')
    ax.set_xticks(iterations)
    ax.set_ylim(80, 100)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'oura_scores.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')

def plot_architecture_diagram():
    """Create a clean architecture diagram with matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layers = [
        (0.5, 8.5, 15, 1.2, '#1E3A5F', 'CORE: Autoentrenamiento Incremental (Google-Style)\nSelf-Training Loop | Aprendizaje Continuo | Synapsis Memory'),
        (0.5, 6.8, 15, 1.2, '#2D5F8A', 'ARQUITECTURAS DE MODELADO\nLLM (Texto) | SNN (Spikes) | SSM (Secuencias) | JEPA (Latente)'),
        (0.5, 5.1, 15, 1.2, '#4A90C4', 'MODELOS .basemateria (Base del Sistema)\nmateria-v3 (678K params) | science-v3 (~1M)'),
        (0.5, 3.4, 15, 1.2, '#6BB3E0', 'MÓDULOS .materia (Expansión de Conocimiento)\nFull (8.1M) | Extended (3.2M) | Unified (2.2M) | Nano (4.1M) | 86+ módulos'),
        (0.5, 1.7, 15, 1.2, '#8CC4F0', 'BACKEND DE INFERENCIA\nOllama | llama.cpp | Cloud API | CPU/GPU'),
    ]

    for x, y, w, h, color, text in layers:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='white', linewidth=2, alpha=0.85, joinstyle='round', linestyle='-')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, color='white', fontweight='bold', linespacing=1.5)

    for i in range(len(layers) - 1):
        _, y1, _, h1, _, _ = layers[i]
        _, y2, _, _, _, _ = layers[i + 1]
        mid_x = 8
        ax.annotate('', xy=(mid_x, y2), xytext=(mid_x, y1 - 0.1),
                     arrowprops=dict(arrowstyle='->', color='#888', lw=2))

    ax.text(8, 0.3, 'Entrada (Prompt/Audio) → Procesamiento → Inferencia → Retroalimentación',
            ha='center', fontsize=11, style='italic', color='#666')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'architecture_diagram.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  {path}')

def plot_model_hierarchy():
    """Model hierarchy as a tree diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    nodes = [
        (7, 7.2, '#1E3A5F', 'MATERIA V3\nSystem'),
        (3.5, 5.5, '#2D5F8A', 'Base\n.basemateria'),
        (10.5, 5.5, '#4A90C4', 'Modules\n.materia'),
        (1.5, 3.8, '#6BB3E0', 'materia-v3\n(678K params)'),
        (5.5, 3.8, '#6BB3E0', 'science-v3\n(~1M params)'),
        (9, 3.8, '#6BB3E0', 'Full\n(8.1M params)'),
        (12, 3.8, '#6BB3E0', 'Extended\n(3.2M params)'),
        (6.5, 2.0, '#8CC4F0', 'Unified\n(2.2M params)'),
        (9.5, 2.0, '#8CC4F0', 'Nano\n(4.1M params)'),
        (3.5, 2.0, '#8CC4F0', '86+ sub-agents\n& expansions'),
    ]

    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (5, 7), (5, 8), (6, 9)]

    for parent, child in edges:
        x1, y1 = nodes[parent][0], nodes[parent][1]
        x2, y2 = nodes[child][0], nodes[child][1]
        ax.plot([x1, x2], [y1 - 0.3, y2 + 0.3], '-', color='#999', linewidth=1.5, alpha=0.6)

    for x, y, color, label in nodes:
        circle = plt.Circle((x, y), 0.55, facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='white', fontweight='bold', linespacing=1.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'model_hierarchy.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  {path}')

def main():
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    all_data = {}
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith('.csv'):
            continue
        name = fname.replace('_log.csv', '')
        path = os.path.join(logs_dir, fname)
        data, steps = read_csv(path)
        all_data[name] = (data, steps)

    print('Generating research plots...')
    plot_loss_comparison(all_data)
    plot_accuracy_comparison(all_data)
    plot_convergence_summary(all_data)
    plot_params_vs_performance(all_data)
    plot_spike_rate_comparison(all_data)
    plot_gradient_norm_comparison(all_data)
    plot_learning_rate(all_data)
    plot_oura_scores()
    plot_architecture_diagram()
    plot_model_hierarchy()

    for key in all_data:
        plot_model_dashboard(all_data, key)

    print(f'\nAll plots saved to {OUTPUT}/')
    print(f'Total: {len(os.listdir(OUTPUT))} files')

if __name__ == '__main__':
    main()
