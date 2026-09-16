"""Train DeCAPS-Net (parsing + skeleton fusion) on GFBMD with 5-fold subject-level CV.

Usage:
    python train_fusion.py --config configs/gfbmd_fusion.yaml
"""

import argparse
import inspect
import json
import os
import pickle
import random
import sys
import time
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            'Class %s cannot be found (%s)'
            % (class_str, traceback.format_exception(*sys.exc_info()))
        )


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------
# Batch / model-output parsing
# ---------------------------------------------------------
def unpack_batch(batch):
    parse_x, skel_x, label, index, meta = batch
    subject_uid = meta.get('subject_uid', None) if isinstance(meta, dict) else None
    return parse_x, skel_x, label, index, subject_uid


def unpack_model_output(output):
    """Returns (fused_logit [B], aux dict with parse_logit/skel_logit)."""
    if torch.is_tensor(output):
        return output.view(-1), {}
    logit, aux = output
    return logit.view(-1), aux


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_metrics_from_logits(y_true, logits, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(logits).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= threshold).astype(int)

    acc = float((pred == y_true).mean())
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, prob))
    except Exception:
        auc = float('nan')
    try:
        ap = float(average_precision_score(y_true, prob))
    except Exception:
        ap = float('nan')

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    return {
        'acc': acc, 'precision': float(p), 'recall': float(r), 'f1': float(f1),
        'auc': auc, 'ap': ap, 'threshold': float(threshold),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'prob': prob, 'pred': pred,
    }


# ---------------------------------------------------------
# Trainer (5-fold subject-level CV)
# ---------------------------------------------------------
class FusionTrainerCV:
    def __init__(self, arg):
        self.arg = arg
        self.device_ids = arg['device'] if isinstance(arg['device'], list) else [arg['device']]
        self.output_device = self.device_ids[0]
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.output_device}' if self.use_cuda else 'cpu')
        self.use_amp = bool(arg.get('amp', False)) and self.use_cuda
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        os.makedirs(arg['work_dir'], exist_ok=True)
        with open(os.path.join(arg['work_dir'], 'config.yaml'), 'w', encoding='utf-8') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg, f, sort_keys=False)

        self.folds_json_path = arg.get('folds_json') or arg['train_feeder_args'].get('folds_json')
        if self.folds_json_path is None:
            raise RuntimeError('Please provide folds_json (top-level or in train_feeder_args).')
        self.folds_list = load_json(self.folds_json_path).get('folds', [])
        if len(self.folds_list) == 0:
            raise RuntimeError("No folds found in folds json.")

        self.monitor_metric = str(arg.get('monitor_metric', 'f1')).lower()
        self.threshold = float(arg.get('threshold', 0.5))
        self.aux_loss_weight = float(arg.get('aux_loss_weight', 0.0))

    def print_log(self, msg, fold_dir=None):
        msg = '[ {} ] {}'.format(time.asctime(time.localtime(time.time())), msg)
        print(msg)
        if self.arg.get('print_log', True):
            log_path = os.path.join(fold_dir if fold_dir else self.arg['work_dir'], 'log.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                print(msg, file=f)

    def _build_feeder_for_fold(self, feeder_cls, base_args, split_name, fold_index):
        args = deepcopy(base_args)
        args['split'] = split_name
        args['fold'] = int(fold_index) + 1  # 0-based -> 1-based
        if 'folds_json' not in args:
            args['folds_json'] = self.folds_json_path
        return feeder_cls(**args)

    def _build_model(self):
        Model = import_class(self.arg['model'])
        return Model(**self.arg['model_args']).to(self.device)

    def _build_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if self.arg['optimizer'] == 'SGD':
            return optim.SGD(params, lr=float(self.arg['base_lr']), momentum=0.9,
                             nesterov=bool(self.arg.get('nesterov', True)),
                             weight_decay=float(self.arg['weight_decay']))
        if self.arg['optimizer'] == 'Adam':
            return optim.Adam(params, lr=float(self.arg['base_lr']),
                              weight_decay=float(self.arg['weight_decay']))
        if self.arg['optimizer'] == 'RMSProp':
            return optim.RMSprop(params, lr=float(self.arg['base_lr']), alpha=0.9,
                                 weight_decay=float(self.arg['weight_decay']))
        raise ValueError(f"Unsupported optimizer: {self.arg['optimizer']}")

    def adjust_learning_rate(self, optimizer, epoch):
        base_lr = float(self.arg['base_lr'])
        warm = int(self.arg['warm_up_epoch'])
        num_epoch_ = int(self.arg['cosine_epoch']) + warm
        if num_epoch_ <= warm:
            num_epoch_ = warm + 1

        if epoch < warm and warm > 0:
            lr = base_lr * (epoch + 1) / warm
        else:
            lr_cos = base_lr * (0.5 * (np.cos((epoch - warm) / (num_epoch_ - warm) * np.pi) + 1))
            if epoch < num_epoch_ and lr_cos > 0.01 * base_lr:
                lr = lr_cos
            else:
                lr = base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg['step'])))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def _criterion(self):
        pw = torch.tensor([float(self.arg.get('loss_pos_weight', 1.0))], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))

    def _save_checkpoint(self, model, optimizer, epoch, metrics, fold_dir, tag):
        ckpt = {
            'epoch': int(epoch),
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
            'model_args': self.arg['model_args'],
        }
        path = os.path.join(fold_dir, f'{tag}.pt')
        torch.save(ckpt, path)
        return path

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.monitor_metric == 'loss':
            cur = float(current.get('loss', float('inf')))
            bst = float(best.get('loss', float('inf')))
            if np.isnan(cur):
                return False
            if np.isnan(bst):
                return True
            return cur <= bst
        cur = float(current.get(self.monitor_metric, float('-inf')))
        bst = float(best.get(self.monitor_metric, float('-inf')))
        if np.isnan(cur):
            return False
        if np.isnan(bst):
            return True
        return cur >= bst

    def _to_device(self, x):
        return x.float().to(self.device, non_blocking=True)

    def _label_to_device(self, y):
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return y.float().view(-1).to(self.device, non_blocking=True)

    # -----------------------------------------------------
    def train_one_epoch(self, epoch, model, loader, optimizer, criterion, fold_dir):
        model.train()
        lr = self.adjust_learning_rate(optimizer, epoch)

        losses, y_true_all, logits_all = [], [], []
        pbar = tqdm(loader, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            parse_x, skel_x, label, index, _ = unpack_batch(batch)
            parse_x = self._to_device(parse_x)
            skel_x = self._to_device(skel_x)
            label = self._label_to_device(label)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda' if self.use_cuda else 'cpu',
                                dtype=torch.float16, enabled=self.use_amp):
                fused_logit, aux = unpack_model_output(model(parse_x, skel_x, return_aux=True))
                loss = criterion(fused_logit, label)
                if self.aux_loss_weight > 0:
                    if aux.get('parse_logit') is not None:
                        loss = loss + self.aux_loss_weight * criterion(aux['parse_logit'].view(-1), label)
                    if aux.get('skel_logit') is not None:
                        loss = loss + self.aux_loss_weight * criterion(aux['skel_logit'].view(-1), label)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            losses.append(float(loss.item()))
            y_true_all.append(label.detach().cpu().numpy().astype(int))
            logits_all.append(fused_logit.detach().cpu().numpy())

            if (batch_idx + 1) % max(1, int(self.arg.get('log_interval', 10))) == 0:
                pbar.set_description(f'Train E{epoch+1} Loss={np.mean(losses):.4f} LR={lr:.6f}')

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
        logits = np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32)
        m = compute_metrics_from_logits(y_true, logits, threshold=self.threshold)
        m['loss'] = float(np.mean(losses)) if losses else float('nan')
        m['lr'] = float(lr)

        self.print_log(f"[train] E{epoch+1} Loss={m['loss']:.4f} Acc={m['acc']*100:.2f}%", fold_dir=fold_dir)
        return m

    @torch.no_grad()
    def eval_one_epoch(self, epoch, model, loader, criterion, fold_dir, save_score=False, prefix='val'):
        model.eval()

        losses, y_true_all, logits_all, subject_uid_all = [], [], [], []
        for batch in tqdm(loader, dynamic_ncols=True):
            parse_x, skel_x, label, index, subject_uid = unpack_batch(batch)
            parse_x = self._to_device(parse_x)
            skel_x = self._to_device(skel_x)
            label_dev = self._label_to_device(label)

            fused_logit, aux = unpack_model_output(model(parse_x, skel_x, return_aux=True))
            loss = criterion(fused_logit, label_dev)
            if self.aux_loss_weight > 0:
                if aux.get('parse_logit') is not None:
                    loss = loss + self.aux_loss_weight * criterion(aux['parse_logit'].view(-1), label_dev)
                if aux.get('skel_logit') is not None:
                    loss = loss + self.aux_loss_weight * criterion(aux['skel_logit'].view(-1), label_dev)

            losses.append(float(loss.item()))
            y_true_all.append(label_dev.cpu().numpy().astype(int))
            logits_all.append(fused_logit.cpu().numpy())
            if subject_uid is not None:
                subject_uid_all.extend([str(x) for x in subject_uid])

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
        logits = np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32)
        m = compute_metrics_from_logits(y_true, logits, threshold=self.threshold)
        m['loss'] = float(np.mean(losses)) if losses else float('nan')

        self.print_log(
            f"[{prefix}] E{epoch+1} Loss={m['loss']:.4f} Acc={m['acc']*100:.2f}% "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"AUC={m['auc']:.4f} AP={m['ap']:.4f} thr={self.threshold}",
            fold_dir=fold_dir)
        self.print_log(
            f"[{prefix}] Confusion (TN FP / FN TP): {m['tn']} {m['fp']} / {m['fn']} {m['tp']}",
            fold_dir=fold_dir)

        if save_score:
            score_out = {
                'y_true': y_true.tolist(),
                'logits': logits.tolist(),
                'prob': m['prob'].tolist(),
                'pred': m['pred'].tolist(),
                'subject_uid': subject_uid_all if len(subject_uid_all) == len(y_true) else [],
            }
            with open(os.path.join(fold_dir, f'{prefix}_scores.pkl'), 'wb') as f:
                pickle.dump(score_out, f)

        return m

    # -----------------------------------------------------
    def run_fold(self, fold_index):
        fold_num = fold_index + 1
        fold_dir = os.path.join(self.arg['work_dir'], f'fold{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)

        self.print_log('=' * 80, fold_dir=fold_dir)
        self.print_log(f'Starting Fold {fold_num}/{len(self.folds_list)}', fold_dir=fold_dir)
        self.print_log('=' * 80, fold_dir=fold_dir)

        Feeder = import_class(self.arg['feeder'])
        train_ds = self._build_feeder_for_fold(Feeder, self.arg['train_feeder_args'], 'train', fold_index)
        val_ds = self._build_feeder_for_fold(Feeder, self.arg['val_feeder_args'], 'val', fold_index)
        self.print_log(f'Train size={len(train_ds)} | Val size={len(val_ds)}', fold_dir=fold_dir)

        loader_kw = dict(num_workers=int(self.arg.get('num_worker', 0)),
                         worker_init_fn=seed_worker, pin_memory=self.use_cuda)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=int(self.arg['batch_size']), shuffle=True, drop_last=False, **loader_kw)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=int(self.arg['test_batch_size']), shuffle=False, drop_last=False, **loader_kw)

        model = self._build_model()
        optimizer = self._build_optimizer(model)
        criterion = self._criterion()

        best_val = None
        best_ckpt_path = None
        per_epoch_rows = []

        for epoch in range(int(self.arg.get('start_epoch', 0)), int(self.arg['num_epoch'])):
            train_metrics = self.train_one_epoch(epoch, model, train_loader, optimizer, criterion, fold_dir)

            do_eval = ((epoch + 1) % int(self.arg.get('eval_interval', 1)) == 0) or ((epoch + 1) == int(self.arg['num_epoch']))
            val_metrics = self.eval_one_epoch(epoch, model, val_loader, criterion, fold_dir) if do_eval else None

            row = {'epoch': int(epoch + 1),
                   'train': {k: v for k, v in train_metrics.items() if k not in ['prob', 'pred']}}
            if val_metrics is not None:
                row['val'] = {k: v for k, v in val_metrics.items() if k not in ['prob', 'pred']}
                current = {k: v for k, v in val_metrics.items() if k not in ['prob', 'pred']}
                if self._is_better(current, best_val):
                    best_val = current
                    best_ckpt_path = self._save_checkpoint(model, optimizer, epoch + 1, current, fold_dir, tag='best')
                    self.print_log(
                        f"[fold {fold_num}] New best ({self.monitor_metric}) = "
                        f"{current.get(self.monitor_metric, np.nan):.4f}", fold_dir=fold_dir)
            per_epoch_rows.append(row)

        last_ckpt_path = self._save_checkpoint(
            model, optimizer, int(self.arg['num_epoch']),
            {k: v for k, v in (val_metrics or train_metrics).items() if k not in ['prob', 'pred']},
            fold_dir, tag='last')

        if best_ckpt_path is None:
            best_ckpt_path = last_ckpt_path

        ckpt = torch.load(best_ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state'])
        best_epoch_num = int(ckpt['epoch'])
        self.print_log(f'Final fold eval with epoch {best_epoch_num} weights', fold_dir=fold_dir)

        best_val_final = self.eval_one_epoch(
            best_epoch_num - 1, model, val_loader, criterion, fold_dir,
            save_score=True, prefix='val_best')

        best_val_clean = {k: v for k, v in best_val_final.items() if k not in ['prob', 'pred']}
        fold_summary = {
            'fold_index_0based': int(fold_index),
            'fold_index_1based': int(fold_num),
            'best_checkpoint': best_ckpt_path,
            'best_epoch': int(best_epoch_num),
            'monitor_metric': self.monitor_metric,
            'best_val_during_training': best_val,
            'final_val_best': best_val_clean,
            'last_checkpoint': last_ckpt_path,
            'num_train_samples': int(len(train_ds)),
            'num_val_samples': int(len(val_ds)),
        }
        save_json(fold_summary, os.path.join(fold_dir, 'fold_summary.json'))
        save_json({'epochs': per_epoch_rows}, os.path.join(fold_dir, 'history.json'))

        self.print_log(f'Fold {fold_num} done.', fold_dir=fold_dir)
        return fold_summary

    def run(self):
        self.print_log(f'Using folds json: {self.folds_json_path}')
        self.print_log(f'Num folds: {len(self.folds_list)}')
        self.print_log(f'Monitor metric: {self.monitor_metric}')
        self.print_log(f'Threshold: {self.threshold}')

        all_fold_summaries = []
        for fi in range(len(self.folds_list)):
            init_seed(int(self.arg.get('seed', 42)) + fi)
            all_fold_summaries.append(self.run_fold(fi))

        metrics_to_agg = ['loss', 'acc', 'precision', 'recall', 'f1', 'auc', 'ap']
        agg = {}
        for metric_name in metrics_to_agg:
            vals = np.asarray([float(fs['final_val_best'].get(metric_name, np.nan))
                               for fs in all_fold_summaries], dtype=float)
            agg[metric_name] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'values': [None if np.isnan(x) else float(x) for x in vals],
            }
        confusion_sum = {
            'tn': int(sum(fs['final_val_best'].get('tn', 0) for fs in all_fold_summaries)),
            'fp': int(sum(fs['final_val_best'].get('fp', 0) for fs in all_fold_summaries)),
            'fn': int(sum(fs['final_val_best'].get('fn', 0) for fs in all_fold_summaries)),
            'tp': int(sum(fs['final_val_best'].get('tp', 0) for fs in all_fold_summaries)),
        }

        cv_summary = {
            'config': {
                'work_dir': self.arg['work_dir'],
                'feeder': self.arg['feeder'],
                'model': self.arg['model'],
                'folds_json': self.folds_json_path,
                'num_folds': len(self.folds_list),
                'seed': int(self.arg.get('seed', 42)),
                'monitor_metric': self.monitor_metric,
                'threshold': self.threshold,
                'loss_pos_weight': float(self.arg.get('loss_pos_weight', 1.0)),
                'aux_loss_weight': self.aux_loss_weight,
            },
            'folds': all_fold_summaries,
            'aggregate': agg,
            'confusion_sum': confusion_sum,
        }
        save_json(cv_summary, os.path.join(self.arg['work_dir'], 'cv_summary.json'))

        self.print_log('=' * 80)
        self.print_log('CV summary (mean over folds):')
        for metric_name in metrics_to_agg:
            self.print_log(
                f"  {metric_name}: {agg[metric_name]['mean']:.4f} +- {agg[metric_name]['std']:.4f}")
        self.print_log(f"  confusion_sum: {confusion_sum}")


def main():
    parser = argparse.ArgumentParser(description='DeCAPS-Net fusion — GFBMD 5-fold CV training')
    parser.add_argument('--config', default='configs/gfbmd_fusion.yaml')
    parser.add_argument('--work-dir', default=None, help='override work_dir from config')
    parser.add_argument('--device', type=int, default=None, help='override device from config')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        arg = yaml.safe_load(f)

    if args.work_dir is not None:
        arg['work_dir'] = args.work_dir
    if args.device is not None:
        arg['device'] = [args.device]

    FusionTrainerCV(arg).run()


if __name__ == '__main__':
    main()
