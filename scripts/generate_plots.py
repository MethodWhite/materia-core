"""
MATERIA V3 - Generador automatico de graficos de entrenamiento
Uso: python scripts/generate_plots.py logs/mi_log.csv --output docs/plots/

Lee cualquier CSV de entrenamiento y genera los mismos graficos
que se usan en el paper cientifico. Funciona headless (sin display).
"""
import os, sys, argparse, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    return data


def generate_plots(csv_path, output_dir, prefix=''):
    os.makedirs(output_dir, exist_ok=True)
    data = read_csv(csv_path)
    steps = list(range(1, len(data[list(data.keys())[0]]) + 1))

    # 1. Loss (train + val)
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'train_loss' in data:
        ax.plot(steps, data['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'val_loss' in data and any(v > 0 for v in data['val_loss']):
        ax.plot(steps, data['val_loss'], 'r--', linewidth=2, label='Val Loss')
    if 'loss' in data:
        ax.plot(steps, data['loss'], 'b-', linewidth=2, label='Loss')
    ax.set_xlabel('Epoch' if len(steps) < 50 else 'Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{prefix}Training Loss', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f'{prefix}loss.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path}")

    # 2. Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'train_acc' in data:
        ax.plot(steps, data['train_acc'], 'b-', linewidth=2, label='Train Acc')
    if 'val_acc' in data and any(v > 0 for v in data['val_acc']):
        ax.plot(steps, data['val_acc'], 'r--', linewidth=2, label='Val Acc')
    if 'accuracy' in data:
        ax.plot(steps, data['accuracy'], 'b-', linewidth=2, label='Accuracy')
    ax.set_xlabel('Epoch' if len(steps) < 50 else 'Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{prefix}Accuracy', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f'{prefix}accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path}")

    # 3. Spike rate (LIF neuron activity)
    for col in ['spike_rate', 'spike']:
        if col in data and any(v > 0 for v in data[col]):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, data[col], 'g-', linewidth=2)
            ax.set_xlabel('Epoch' if len(steps) < 50 else 'Step', fontsize=12)
            ax.set_ylabel('Spike Rate', fontsize=12)
            ax.set_title(f'{prefix}LIF Neuron Spike Rate', fontsize=14)
            ax.grid(True, alpha=0.3)
            path = os.path.join(output_dir, f'{prefix}spike_rate.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  {path}")
            break

    # 4. Gradient norm
    for col in ['grad_norm', 'gradient_norm']:
        if col in data:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, data[col], 'orange', linewidth=1.5)
            ax.set_xlabel('Epoch' if len(steps) < 50 else 'Step', fontsize=12)
            ax.set_ylabel('Gradient Norm', fontsize=12)
            ax.set_title(f'{prefix}Gradient Norm (Training Stability)', fontsize=14)
            ax.grid(True, alpha=0.3)
            path = os.path.join(output_dir, f'{prefix}grad_norm.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  {path}")
            break

    # 5. Learning rate
    if 'lr' in data and len(set(data['lr'])) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, data['lr'], 'm-', linewidth=2)
        ax.set_xlabel('Epoch' if len(steps) < 50 else 'Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'{prefix}Learning Rate Schedule', fontsize=14)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, f'{prefix}learning_rate.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {path}")

    # 6. Combined dashboard
    n_plots = sum([
        1,  # loss
        1,  # accuracy
        1 if any(c in data for c in ['spike_rate', 'spike']) else 0,
        1 if any(c in data for c in ['grad_norm', 'gradient_norm']) else 0,
    ])
    if n_plots >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        if 'train_loss' in data:
            ax.plot(steps, data['train_loss'], 'b-', label='Train')
        if 'val_loss' in data and any(v > 0 for v in data['val_loss']):
            ax.plot(steps, data['val_loss'], 'r--', label='Val')
        if 'loss' in data:
            ax.plot(steps, data['loss'], 'b-', label='Loss')
        ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if 'train_acc' in data:
            ax.plot(steps, data['train_acc'], 'b-', label='Train')
        if 'val_acc' in data and any(v > 0 for v in data['val_acc']):
            ax.plot(steps, data['val_acc'], 'r--', label='Val')
        if 'accuracy' in data:
            ax.plot(steps, data['accuracy'], 'b-', label='Acc')
        ax.set_title('Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for col in ['spike_rate', 'spike']:
            if col in data and any(v > 0 for v in data[col]):
                ax.plot(steps, data[col], 'g-')
                ax.set_title('Spike Rate (LIF)'); ax.grid(True, alpha=0.3)
                break

        ax = axes[1, 1]
        for col in ['grad_norm', 'gradient_norm']:
            if col in data:
                ax.plot(steps, data[col], 'orange')
                ax.set_title('Gradient Norm'); ax.grid(True, alpha=0.3)
                break

        plt.tight_layout()
        path = os.path.join(output_dir, f'{prefix}training_curves.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {path}")

    print(f"Done: {output_dir}")


def batch_generate(logs_dir, output_dir):
    """Generate plots for all CSVs in a directory."""
    for fname in sorted(os.listdir(logs_dir)):
        if fname.endswith('.csv'):
            name = fname.replace('.csv', '').replace('_log', '')
            csv_path = os.path.join(logs_dir, fname)
            out = os.path.join(output_dir, name)
            print(f"\n{name}:")
            generate_plots(csv_path, out, prefix=f'{name}_')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='CSV log file or logs directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: same dir as input)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all CSVs in input directory')
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        out = args.output or os.path.join(args.input, '..', 'plots')
        batch_generate(args.input, os.path.abspath(out))
    else:
        out = args.output or os.path.join(os.path.dirname(args.input), 'plots')
        name = os.path.splitext(os.path.basename(args.input))[0]
        generate_plots(args.input, os.path.abspath(out), prefix=f'{name}_')
