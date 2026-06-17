"""
Oura Loop Engine — Orquestrador de Entrenamiento para M.A.T.E.R.I.A. V3
Integra: loop de iteraciones, detección de convergencia, feedback multi-fuente
"""
import os, json, time, math, csv, sys
import numpy as np

class OuraLoopEngine:
    def __init__(self, model, train_loader, val_loader, model_name="model",
                 max_iterations=20, convergence_threshold=90.0,
                 patience=3, min_delta=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_delta = min_delta

        self.history = {'iteration': [], 'loss': [], 'accuracy': [],
                        'val_loss': [], 'val_accuracy': [], 'score': [],
                        'grad_norm': [], 'learning_rate': [], 'spike_rate': [],
                        'converged': []}
        self.best_metrics = {'loss': float('inf'), 'accuracy': 0.0,
                            'val_loss': float('inf'), 'val_accuracy': 0.0}
        self.plateau_count = 0
        self.loop_id = f"oura_{model_name}_{int(time.time())}"
        self.start_time = time.time()

    def compute_score(self, loss, accuracy, val_loss, val_accuracy):
        """Score compuesto 0-100: balance entre loss bajo y accuracy alta"""
        loss_score = max(0, 100 - (loss * 100))
        acc_score = accuracy * 100
        val_loss_score = max(0, 100 - (val_loss * 100))
        val_acc_score = val_accuracy * 100
        # Weighted: 30% train loss, 30% train acc, 20% val loss, 20% val acc
        score = (loss_score * 0.3) + (acc_score * 0.3) + \
                (val_loss_score * 0.2) + (val_acc_score * 0.2)
        return min(100, max(0, score))

    def check_convergence(self, current_loss, current_acc):
        """Detecta convergencia: plateau en loss y accuracy"""
        if len(self.history['loss']) < 2:
            return False

        # Loss plateau: mejora < min_delta por patience iteraciones
        recent_losses = self.history['loss'][-self.patience:]
        loss_improvement = recent_losses[0] - recent_losses[-1]

        if abs(loss_improvement) < self.min_delta and current_acc > 0.95:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        return self.plateau_count >= self.patience

    def iteration(self, opt, sch, epoch, total_epochs):
        """Una iteración del loop Oura"""
        self.model.train()
        total_loss, total_acc, total_steps = 0.0, 0.0, len(self.train_loader)

        import torch
        import torch.nn.functional as F

        for i, (x, y) in enumerate(self.train_loader):
            opt.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sch.step()

            preds = logits.argmax(dim=-1)
            mask = y != 0
            correct = (preds[mask] == y[mask]).float().sum()
            acc = (correct / mask.sum()).item() if mask.sum() > 0 else 0.0

            total_loss += loss.item()
            total_acc += acc

            if (i+1) % 50 == 0:
                sr = 0.0
                if hasattr(self.model, 'snn') and self.model.use_snn:
                    with torch.no_grad():
                        _, sr = self.model.snn(self.model.tok_emb(x[:, -1:]))
                        sr = sr.item() if torch.is_tensor(sr) else sr
                print(f"  [{self.loop_id[:8]}] E{epoch+1}/{total_epochs} [{i+1}/{total_steps}] "
                      f"loss={loss.item():.4f} acc={acc:.4f} gn={grad_norm:.2f} sr={sr:.3f}")

        avg_loss = total_loss / total_steps
        avg_acc = total_acc / total_steps

        # Validation
        self.model.eval()
        val_loss, val_acc, val_steps = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                mask = y != 0
                correct = (preds[mask] == y[mask]).float().sum()
                val_acc += (correct / mask.sum()).item() if mask.sum() > 0 else 0.0
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_acc = val_acc / max(1, val_steps)

        # Compute score
        score = self.compute_score(avg_loss, avg_acc, avg_val_loss, avg_val_acc)

        # Track history
        self.history['iteration'].append(epoch)
        self.history['loss'].append(avg_loss)
        self.history['accuracy'].append(avg_acc)
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_accuracy'].append(avg_val_acc)
        self.history['score'].append(score)
        self.history['grad_norm'].append(grad_norm if isinstance(grad_norm, float) else grad_norm.item())
        self.history['learning_rate'].append(sch.get_last_lr()[0])
        self.history['spike_rate'].append(0.0)

        # Update best
        if avg_loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = avg_loss
            self.best_metrics['accuracy'] = avg_acc
            self.best_metrics['val_loss'] = avg_val_loss
            self.best_metrics['val_accuracy'] = avg_val_acc

        # Convergence check
        converged = self.check_convergence(avg_loss, avg_acc)
        self.history['converged'].append(converged)

        print(f"  [{self.loop_id[:8]}] → E{epoch+1}: train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} "
              f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f} score={score:.1f} "
              f"{'(CONVERGED)' if converged else ''}")

        return avg_loss, avg_acc, score, converged

    def run(self, opt, sch, total_epochs, csv_path=None):
        """Ejecuta el loop completo de Oura"""
        import torch

        print(f"\n{'='*60}")
        print(f"OURA LOOP ENGINE — {self.model_name}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Convergence threshold: {self.convergence_threshold}")
        print(f"  Patience: {self.patience}")
        print(f"  Loop ID: {self.loop_id}")
        print(f"{'='*60}")

        csv_file = None
        csv_writer = None
        if csv_path:
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['iteration', 'loss', 'accuracy', 'val_loss', 'val_accuracy',
                                'score', 'grad_norm', 'learning_rate', 'spike_rate', 'converged'])

        for epoch in range(min(total_epochs, self.max_iterations)):
            loss, acc, score, converged = self.iteration(opt, sch, epoch, total_epochs)

            if csv_writer:
                csv_writer.writerow([epoch+1, f'{loss:.4f}', f'{acc:.4f}',
                                    f'{self.history["val_loss"][-1]:.4f}',
                                    f'{self.history["val_accuracy"][-1]:.4f}',
                                    f'{score:.1f}',
                                    f'{self.history["grad_norm"][-1]:.4f}',
                                    f'{self.history["learning_rate"][-1]:.2e}',
                                    f'{self.history["spike_rate"][-1]:.3f}',
                                    converged])

            # Convergence-based early stopping
            if converged and score >= self.convergence_threshold:
                print(f"\n  ✓ Oura convergió en iteración {epoch+1}")
                print(f"    Score: {score:.1f}/{self.convergence_threshold}")
                print(f"    Loss: {loss:.4f} | Acc: {acc:.4f}")
                break

        if csv_file:
            csv_file.close()

        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"OURA LOOP COMPLETE — {self.model_name}")
        print(f"  Iterations: {epoch+1}/{total_epochs}")
        print(f"  Best loss: {self.best_metrics['loss']:.4f}")
        print(f"  Best acc: {self.best_metrics['accuracy']:.4f}")
        print(f"  Best val_loss: {self.best_metrics['val_loss']:.4f}")
        print(f"  Best val_acc: {self.best_metrics['val_accuracy']:.4f}")
        print(f"  Final score: {self.history['score'][-1]:.1f}/100")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'='*60}")

        return self.best_metrics

    def generate_report(self, output_dir):
        """Genera reporte de entrenamiento con gráficos Oura-style"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Oura Loop Report: {self.model_name} (ID: {self.loop_id})', fontsize=14)

        iters = self.history['iteration']

        # Loss
        axes[0,0].plot(iters, self.history['loss'], 'b-', label='Train')
        axes[0,0].plot(iters, self.history['val_loss'], 'b--', label='Val')
        axes[0,0].set_title('Loss'); axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

        # Accuracy
        axes[0,1].plot(iters, self.history['accuracy'], 'g-', label='Train')
        axes[0,1].plot(iters, self.history['val_accuracy'], 'g--', label='Val')
        axes[0,1].set_title('Accuracy'); axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()

        # Score
        axes[0,2].plot(iters, self.history['score'], 'r-', linewidth=2)
        axes[0,2].axhline(y=self.convergence_threshold, color='gray', linestyle='--', alpha=0.5, label=f'Threshold={self.convergence_threshold}')
        axes[0,2].set_title('Oura Score'); axes[0,2].grid(True, alpha=0.3); axes[0,2].legend()

        # Gradient norm
        axes[1,0].plot(iters, self.history['grad_norm'], 'purple')
        axes[1,0].set_title('Gradient Norm'); axes[1,0].grid(True, alpha=0.3)

        # Learning rate
        axes[1,1].plot(iters, self.history['learning_rate'], 'orange')
        axes[1,1].set_title('Learning Rate'); axes[1,1].grid(True, alpha=0.3)

        # Convergence marker
        conv = [1 if c else 0 for c in self.history['converged']]
        axes[1,2].fill_between(iters, 0, conv, alpha=0.5, color='green', label='Converged')
        axes[1,2].set_title('Convergence'); axes[1,2].grid(True, alpha=0.3); axes[1,2].set_ylim(-0.1, 1.1)

        plt.tight_layout()
        path = os.path.join(output_dir, f'oura_{self.model_name}_report.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Oura report saved: {path}")

        return path
