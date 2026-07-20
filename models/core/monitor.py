"""
Monitor inteligente para MATERIA V4
- Auto-checkpoint por batches
- Deteccion de sobreentrenamiento
- Extraccion de estado interno
- Reporte de aprendizaje
"""
import os, json, time, pickle, gc
import numpy as np
import torch

log = lambda msg: print(f"[MON] {msg}", flush=True)


class ActivationHook:
    """Hook para capturar activaciones intermedias sin modificarr el forward."""
    def __init__(self):
        self.acts = {}

    def hook_fn(self, name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                self.acts[name] = out[0].detach().cpu().float()
            else:
                self.acts[name] = out.detach().cpu().float()
        return fn

    def clear(self):
        self.acts = {}


class TrainingMonitor:
    def __init__(self, model, output_dir, cfg, patience=2, plateau_thresh=1e-4):
        self.model = model
        self.output_dir = output_dir
        self.cfg = cfg
        self.patience = patience
        self.plateau_thresh = plateau_thresh
        self.no_improve = 0

        self.hook = ActivationHook()
        self._register_activation_hooks(model)

        self.stats = {
            'train_loss': [], 'train_tok': [], 'train_jepa': [], 'train_acc': [],
            'val_loss': [], 'val_jepa': [], 'val_acc': [],
            'spike_rate': [], 'spectral_mu': [], 'lr': [],
            'overfit_gap': [], 'perplexity': [],
            'latent_entropy': [], 'synapsis_usage': [],
            'batch_checkpoints': [],
        }
        self.batch_count = 0
        self.best_val_loss = float('inf')

    def _register_activation_hooks(self, model):
        for name, module in model.named_modules():
            if 'snn' in name or 'ssm' in name or 'jepa_enc' in name or 'jepa_pred' in name:
                module.register_forward_hook(self.hook.hook_fn(name))
            if 'synapsis' in name:
                module.register_forward_hook(self.hook.hook_fn(name))

    def check_overfit(self, train_loss, val_loss):
        gap = train_loss - val_loss
        overfit_score = max(0, -gap)
        self.stats['overfit_gap'].append(gap)
        is_overfitting = overfit_score > 0.2
        if is_overfitting:
            log(f"[ROJO] Senal de sobreentrenamiento detectada! gap={gap:.4f}")
        return overfit_score

    def check_plateau(self, val_loss):
        if val_loss < self.best_val_loss - self.plateau_thresh:
            self.best_val_loss = val_loss
            self.no_improve = 0
            return False
        self.no_improve += 1
        if self.no_improve >= self.patience:
            log(f"[INFO] Estancamiento detectado: {self.patience} epochs sin mejora")
            return True
        return False

    def extract_internal_state(self):
        report = {}
        with torch.no_grad():
            if hasattr(self.model, 'jepa_pred') and hasattr(self.model.jepa_pred, 'mu'):
                mu = self.model.jepa_pred.mu.sigmoid()
                report['spectral_mu_mean'] = mu.mean().item()
                report['spectral_mu_std'] = mu.std().item()
                report['spectral_mu_range'] = (mu.min().item(), mu.max().item())

            spike_rate = getattr(self.model, 'spike_rate', 0)
            if hasattr(self.model, 'snn'):
                w_in = getattr(self.model.snn.w_in, 'weight', None)
                if w_in is not None:
                    w = w_in.detach().float()
                    report['snn_weight_mean'] = w.mean().item()
                    report['snn_weight_sparsity'] = (w.abs() < 1e-4).float().mean().item()
            report['spike_rate'] = spike_rate
            report['synapsis_usage'] = self._get_synapsis_usage()
            report['jepa_latent_norm'] = self._get_jepa_latent_norm()

        perf = self._get_performance_summary()
        report.update(perf)
        return report

    def _get_synapsis_usage(self):
        for name, module in self.model.named_modules():
            if 'synapsis' in name and hasattr(module, 'step'):
                total_slots = getattr(module, 'n_slots', 128)
                step = module.step.item()
                filled = min(100.0, step / total_slots * 100)
                return filled
        return 0.0

    def _get_jepa_latent_norm(self):
        if 'jepa_enc' in self.hook.acts:
            latent = self.hook.acts['jepa_enc']
            return latent.norm(dim=-1).mean().item()
        return 0.0

    def _get_performance_summary(self):
        if len(self.stats['train_loss']) < 2:
            return {'convergence_rate': 0, 'memorization_score': 0}

        recent_loss = self.stats['train_loss'][-3:]
        loss_lr = recent_loss[-1] / max(recent_loss[0], 1e-8)
        convergence = max(0, 1.0 - loss_lr)

        if self.stats.get('val_loss') and len(self.stats['val_loss']) > 1:
            val_trend = self.stats['val_loss'][-1] / max(self.stats['val_loss'][0], 1e-8)
            memorization = max(0, val_trend - loss_lr) if val_trend > loss_lr else 0
        else:
            memorization = 0

        return {'convergence_rate': convergence, 'memorization_score': memorization}

    def log_epoch(self, epoch, metrics):
        self.stats['train_loss'].append(metrics.get('train_loss', 0))
        self.stats['train_tok'].append(metrics.get('train_tok', 0))
        self.stats['train_jepa'].append(metrics.get('train_jepa', 0))
        self.stats['train_acc'].append(metrics.get('train_acc', 0))
        self.stats['val_loss'].append(metrics.get('val_loss', 0))
        self.stats['val_jepa'].append(metrics.get('val_jepa', 0))
        self.stats['val_acc'].append(metrics.get('val_acc', 0))
        self.stats['spike_rate'].append(metrics.get('spike_rate', 0))
        self.stats['spectral_mu'].append(metrics.get('spectral_mu', 0))
        self.stats['lr'].append(metrics.get('lr', 0))

        val_loss = metrics.get('val_loss', 0)
        train_loss = metrics.get('train_loss', 0)
        ppl = np.exp(min(val_loss, 20))
        self.stats['perplexity'].append(ppl)
        overfit = self.check_overfit(train_loss, val_loss)

        internal = self.extract_internal_state()
        log(f"  [MON] PPL={ppl:.2f} | overfit={overfit:.4f} | "
            f"spike={internal.get('spike_rate', 0):.4f} | "
            f"synapsis={internal.get('synapsis_usage', 0):.1f}% | "
            f"spectral mu={internal.get('spectral_mu_mean', 0):.4f}")
        if internal.get('convergence_rate', 0) > 0.9:
            log(f"  [VERDE] Alta convergencia: {internal['convergence_rate']:.2%}")
        if internal.get('memorization_score', 0) > 0.3:
            log(f"  [AMARILLO] Posible memorizacion: score={internal['memorization_score']:.3f}")

        return self.check_plateau(val_loss)

    def log_batch(self, metrics):
        self.batch_count += 1
        if self.batch_count % 500 == 0:
            self._emergency_checkpoint(metrics)
            self.stats['batch_checkpoints'].append({
                'batch': self.batch_count,
                'loss': metrics.get('loss', 0),
                'acc': metrics.get('acc', 0),
            })

    def _emergency_checkpoint(self, metrics):
        path = os.path.join(self.output_dir, f'batch_ckpt_{self.batch_count}.pt')
        model_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save({
            'batch': self.batch_count,
            'model_state_dict': model_cpu,
            'metrics': metrics,
        }, path)
        del model_cpu
        log(f"[CHECKPOINT] Batch {self.batch_count} guardado ({os.path.getsize(path)//1024**2}MB)")

    def generate_report(self):
        report = {
            'model': self.cfg.get('name', 'materia-v4'),
            'params': sum(p.numel() for p in self.model.parameters()),
            'epochs_completed': len(self.stats['train_loss']),
            'batches_processed': self.batch_count,
            'final_metrics': {
                'train_loss': float(self.stats['train_loss'][-1]) if self.stats['train_loss'] else 0,
                'val_loss': float(self.stats['val_loss'][-1]) if self.stats['val_loss'] else 0,
                'val_acc': float(self.stats['val_acc'][-1]) if self.stats['val_acc'] else 0,
                'perplexity': float(self.stats['perplexity'][-1]) if self.stats['perplexity'] else 0,
            },
            'learning_status': self._assess_learning(),
            'internal_state': self.extract_internal_state(),
        }
        path = os.path.join(self.output_dir, 'training_report.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        log(f"[REPORTE] Guardado: {path}")
        return report

    def _assess_learning(self):
        status = {'phase': 'unknown', 'needs_more_data': True, 'reason': ''}

        if len(self.stats['train_loss']) < 2:
            status['phase'] = 'calentamiento'
            status['reason'] = 'Menos de 2 epochs completadas'
            return status

        recent_val = np.mean(self.stats['val_loss'][-2:]) if len(self.stats['val_loss']) >= 2 else self.stats['val_loss'][-1]
        initial_val = self.stats['val_loss'][0]
        improvement = (initial_val - recent_val) / max(initial_val, 1e-8)

        convergence = self.stats.get('convergence_rate', [0])[-1] if isinstance(self.stats.get('convergence_rate'), list) else 0
        memorization = self.stats.get('memorization_score', [0])[-1] if isinstance(self.stats.get('memorization_score'), list) else 0

        if improvement < 0.05 and len(self.stats['train_loss']) >= 3:
            if memorization > 0.3:
                status['phase'] = 'sobreentrenado'
                status['needs_more_data'] = True
                status['reason'] = f'Mejora minima ({improvement:.1%}), memorizando (score={memorization:.2f})'
                status['suggestion'] = 'Agregar datos nuevos y variados, reducir epochs'
            else:
                status['phase'] = 'estancado'
                status['needs_more_data'] = True
                status['reason'] = f'Mejora minima ({improvement:.1%}), modelo estabilizado'
                status['suggestion'] = 'Aumentar capacidad del modelo o agregar datos'
        elif improvement > 0.3:
            status['phase'] = 'aprendiendo'
            status['needs_more_data'] = False
            status['reason'] = f'Mejora significativa ({improvement:.1%}), aun aprendiendo'
            status['suggestion'] = 'Continuar training con mismo dataset'
        else:
            status['phase'] = 'estable'
            status['needs_more_data'] = improvement < 0.15
            status['reason'] = f'Mejora moderada ({improvement:.1%})'
            status['suggestion'] = 'Evaluar si agregar mas datos o aumentar modelo'

        return status

    def get_summary_line(self):
        if not self.stats['train_loss']:
            return "Sin datos"
        internal = self.extract_internal_state()
        phase = self._assess_learning()['phase']
        gap = self.stats['overfit_gap'][-1] if self.stats['overfit_gap'] else 0
        tag = 'SOBREENTRENADO' if gap < -0.2 else 'APRENDIENDO' if phase == 'aprendiendo' else 'ESTABLE'
        return (f"[{tag}] loss={self.stats['val_loss'][-1]:.4f} "
                f"ppl={self.stats['perplexity'][-1]:.1f} "
                f"gap={gap:.3f} spike={internal['spike_rate']:.3f} "
                f"syn={internal['synapsis_usage']:.0f}% "
                f"fase={phase}")
