"""Train the DeCAPS-Net skeleton branch on ASDPose (SMM recognition).

Usage:
    python train_asdpose.py --config configs/asdpose_skeleton.yaml
    python train_asdpose.py --config configs/asdpose_skeleton.yaml --phase test --weights work_dir/asdpose_skeleton/best.pt
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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


# ---------------------------------------------------------
# Metrics / aggregation helpers
# ---------------------------------------------------------
def softmax_np(logits):
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)


def binary_metrics(score_eval, y_true_eval, threshold):
    """Metrics from (N, 2) logits; positive class = 1."""
    probs = softmax_np(score_eval)
    y_score_pos = probs[:, 1]
    y_pred = (y_score_pos >= threshold).astype(int)

    acc = float((y_pred == y_true_eval).mean())
    prec = precision_score(y_true_eval, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_true_eval, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_true_eval, y_pred, average='binary', pos_label=1, zero_division=0)

    auc = float('nan')
    ap = float('nan')
    if len(np.unique(y_true_eval)) == 2:
        try:
            auc = float(roc_auc_score(y_true_eval, y_score_pos))
        except Exception:
            pass
        try:
            ap = float(average_precision_score(y_true_eval, y_score_pos))
        except Exception:
            pass

    cm = confusion_matrix(y_true_eval, y_pred, labels=[0, 1])
    return {
        'acc': acc, 'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
        'auc': auc, 'ap': ap, 'cm': cm, 'y_pred': y_pred, 'probs': probs,
    }


def aggregate_windows_to_clips(score_win, y_true_win, idx_win, metas_win):
    """Mean of window logits per clip (grouped by global row index)."""
    clip_map = OrderedDict()
    for n in range(len(y_true_win)):
        m = metas_win[n]
        if m is not None and ('row_index_global' in m):
            clip_key = int(m['row_index_global'])
        elif m is not None and ('identifier' in m):
            clip_key = str(m['identifier'])
        else:
            clip_key = int(idx_win[n])

        if clip_key not in clip_map:
            clip_map[clip_key] = {'logits_list': [], 'y_true': int(y_true_win[n]), 'meta': m}
        clip_map[clip_key]['logits_list'].append(score_win[n].astype(np.float32))

    score_clip, y_true_clip, idx_clip, metas_clip, nwin_clip = [], [], [], [], []
    for k, rec in clip_map.items():
        win_logits = np.stack(rec['logits_list'], axis=0)
        score_clip.append(win_logits.mean(axis=0).astype(np.float32))
        y_true_clip.append(rec['y_true'])
        metas_clip.append(rec['meta'])
        nwin_clip.append(len(rec['logits_list']))
        m = rec['meta']
        idx_clip.append(int(m['row_index_global']) if m is not None and 'row_index_global' in m else -1)

    return (
        np.asarray(score_clip, dtype=np.float32),
        np.asarray(y_true_clip, dtype=np.int64),
        np.asarray(idx_clip, dtype=np.int64),
        metas_clip,
        np.asarray(nwin_clip, dtype=np.int32),
    )


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------
class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.work_dir = arg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)

        with open(os.path.join(self.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg, f, sort_keys=False)

        self.device_ids = arg['device'] if isinstance(arg['device'], list) else [arg['device']]
        self.output_device = self.device_ids[0]
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.output_device}' if self.use_cuda else 'cpu')

        self.best_metric_name = str(arg.get('best_metric', 'f1')).lower()
        if self.best_metric_name not in ['acc', 'precision', 'recall', 'f1']:
            self.best_metric_name = 'f1'
        self.threshold = float(arg.get('decision_threshold', 0.5))

        self.best_metric_value = -1.0
        self.best_metric_epoch = 0
        self.best_metric_snapshot = {}

        self.load_data()
        self.load_model()
        self.load_optimizer()

    def print_log(self, msg):
        msg = '[ {} ] {}'.format(time.asctime(time.localtime(time.time())), msg)
        print(msg)
        if self.arg.get('print_log', True):
            with open(os.path.join(self.work_dir, 'log.txt'), 'a', encoding='utf-8') as f:
                print(msg, file=f)

    def load_data(self):
        Feeder = import_class(self.arg['feeder'])
        common = dict(num_workers=int(self.arg.get('num_worker', 0)), pin_memory=True)
        if common['num_workers'] > 0:
            common['persistent_workers'] = True
            common['prefetch_factor'] = 4

        train_dataset = Feeder(**self.arg['train_feeder_args'])
        self.data_loader = {
            'train': torch.utils.data.DataLoader(
                train_dataset,
                batch_size=int(self.arg['batch_size']),
                shuffle=True,
                drop_last=True,
                worker_init_fn=seed_worker,
                **common,
            ),
            'test': torch.utils.data.DataLoader(
                Feeder(**self.arg['test_feeder_args']),
                batch_size=int(self.arg['test_batch_size']),
                shuffle=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                **common,
            ),
        }

    def load_model(self):
        Model = import_class(self.arg['model'])
        self.model = Model(**self.arg['model_args']).to(self.device)

        pos_weight = float(self.arg.get('loss_pos_weight', 1.0))
        class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=self.device)
        self.loss = nn.CrossEntropyLoss(weight=class_weight)

    def load_optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=float(self.arg['base_lr']),
            momentum=0.9,
            nesterov=bool(self.arg.get('nesterov', True)),
            weight_decay=float(self.arg['weight_decay']),
        )

    def adjust_learning_rate(self, epoch):
        base_lr = float(self.arg['base_lr'])
        warm = int(self.arg['warm_up_epoch'])
        num_epoch_ = int(self.arg['cosine_epoch']) + warm
        if epoch < warm:
            lr = base_lr * (epoch + 1) / warm
        elif epoch < num_epoch_:
            lr = base_lr * (0.5 * (np.cos((epoch - warm) / (num_epoch_ - warm) * np.pi) + 1))
        else:
            lr = base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg['step'])))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def save_checkpoint(self, epoch, metrics, tag):
        ckpt = {
            'epoch': int(epoch),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_args': self.arg['model_args'],
        }
        torch.save(ckpt, os.path.join(self.work_dir, f'{tag}.pt'))

    # -----------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        lr = self.adjust_learning_rate(epoch)
        self.print_log(f'Training epoch: {epoch + 1} (lr={lr:.6f})')

        loss_value, acc_value = [], []
        for data, label, index in tqdm(self.data_loader['train'], dynamic_ncols=True):
            data = data.to(self.device).float()
            label = label.to(self.device).long()

            output = self.model(data)
            loss = sum([self.loss(out, label) for out in output])
            output = sum(output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            predict_label = torch.max(output.data, 1)[1]
            acc_value.append(torch.mean((predict_label == label.data).float()).data.item())

        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(
                np.mean(loss_value), np.mean(acc_value) * 100)
        )

    @torch.no_grad()
    def eval_epoch(self, epoch, save_score=False):
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')

        score_frag, y_true_frag, idx_frag, loss_value = [], [], [], []
        for data, label, index in tqdm(self.data_loader['test'], ncols=60):
            data = data.to(self.device).float()
            label_cuda = label.to(self.device).long()

            output = self.model(data)
            loss = sum([self.loss(out, label_cuda) for out in output])
            logits = sum(output)

            score_frag.append(logits.detach().cpu().numpy())
            y_true_frag.append(label.cpu().numpy() if torch.is_tensor(label) else np.asarray(label))
            idx_frag.append(index.cpu().numpy() if torch.is_tensor(index) else np.asarray(index))
            loss_value.append(loss.data.item())

        score_win = np.concatenate(score_frag, axis=0).astype(np.float32)
        y_true_win = np.concatenate(y_true_frag, axis=0).astype(np.int64)
        idx_win = np.concatenate(idx_frag, axis=0).astype(np.int64)
        loss = float(np.mean(loss_value)) if loss_value else 0.0

        ds = self.data_loader['test'].dataset
        metas_win = [ds.get_meta(int(i)) for i in idx_win]

        use_clip_aggregation = bool(
            getattr(ds, 'split', None) == 'test' and getattr(ds, 'test_multi_windows', False)
        )
        if use_clip_aggregation:
            score_eval, y_true_eval, idx_eval, metas_eval, nwin_eval = aggregate_windows_to_clips(
                score_win, y_true_win, idx_win, metas_win)
        else:
            score_eval, y_true_eval, idx_eval, metas_eval, nwin_eval = (
                score_win, y_true_win, idx_win, metas_win, None)

        m = binary_metrics(score_eval, y_true_eval, self.threshold)
        metric_map = {'acc': m['acc'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1']}
        sel_val = metric_map[self.best_metric_name]

        if sel_val > self.best_metric_value:
            self.best_metric_value = sel_val
            self.best_metric_epoch = epoch + 1
            self.best_metric_snapshot = {
                'epoch': epoch + 1, 'acc': m['acc'], 'precision': m['precision'],
                'recall': m['recall'], 'f1': m['f1'], 'auc': m['auc'], 'ap': m['ap'],
                'threshold': self.threshold,
            }
            self.save_checkpoint(epoch + 1, self.best_metric_snapshot, tag='best')

        self.print_log(
            f'[test] Loss={loss:.4f}  Acc={m["acc"]*100:.2f}%  '
            f'P={m["precision"]:.4f}  R={m["recall"]:.4f}  F1={m["f1"]:.4f}  '
            f'AUC={m["auc"]:.4f}  AP={m["ap"]:.4f}  thr={self.threshold:.3f}'
        )
        tn, fp, fn, tp = m['cm'].ravel()
        self.print_log(f'[test] Confusion (TN FP / FN TP): {tn} {fp} / {fn} {tp}')

        metrics_path = os.path.join(self.work_dir, 'test_metrics.csv')
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'loss', 'acc', 'precision', 'recall', 'f1', 'auc', 'ap', 'threshold'])
            writer.writerow([epoch + 1, loss, m['acc'], m['precision'], m['recall'],
                             m['f1'], m['auc'], m['ap'], self.threshold])

        if save_score:
            probs = m['probs']
            y_pred = m['y_pred']

            score_dict = OrderedDict()
            for n in range(len(score_eval)):
                sid = metas_eval[n].get('identifier', str(int(idx_eval[n]))) if metas_eval[n] else str(int(idx_eval[n]))
                score_dict[sid] = score_eval[n]
            with open(os.path.join(self.work_dir, 'test_score.pkl'), 'wb') as f:
                pickle.dump(score_dict, f)

            with open(os.path.join(self.work_dir, 'test_predictions.csv'), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'clip_row_index_global', 'identifier', 'child_id', 'assessment_id', 'video_id',
                    'action_name', 'event_start_frame_global', 'event_end_frame_global',
                    'clip_len_raw', 'fps', 'num_windows_aggregated',
                    'y_true', 'y_pred', 'prob_0', 'prob_1', 'logit_0', 'logit_1',
                ])
                for n in range(len(y_true_eval)):
                    mt = metas_eval[n] or {}
                    nw = int(nwin_eval[n]) if nwin_eval is not None else 1
                    writer.writerow([
                        int(idx_eval[n]), mt.get('identifier', ''), mt.get('child_id', ''),
                        mt.get('assessment_id', ''), mt.get('video_id', ''), mt.get('action_name', ''),
                        mt.get('event_start_frame_global', ''), mt.get('event_end_frame_global', ''),
                        mt.get('clip_len_raw', ''), mt.get('fps', ''), nw,
                        int(y_true_eval[n]), int(y_pred[n]),
                        float(probs[n, 0]), float(probs[n, 1]),
                        float(score_eval[n, 0]), float(score_eval[n, 1]),
                    ])

            cm = m['cm']
            each_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-12)
            with open(os.path.join(self.work_dir, 'test_confusion.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc.tolist())
                writer.writerows(cm.tolist())

        return {'loss': loss, **metric_map, 'auc': m['auc'], 'ap': m['ap']}

    # -----------------------------------------------------
    def start(self):
        if self.arg.get('phase', 'train') == 'test':
            weights = self.arg.get('weights', None)
            if not weights:
                raise ValueError('Please provide --weights for phase=test.')
            ckpt = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
            self.print_log(f'Loaded weights: {weights}')
            self.eval_epoch(epoch=int(ckpt.get('epoch', 1)) - 1 if isinstance(ckpt, dict) else 0,
                            save_score=self.arg.get('save_score', True))
            return

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Model params: {num_params}')

        for epoch in range(int(self.arg.get('start_epoch', 0)), int(self.arg['num_epoch'])):
            self.print_log('*' * 100)
            self.train_epoch(epoch)
            self.save_checkpoint(epoch + 1, {}, tag='last')

            if ((epoch + 1) % int(self.arg.get('eval_interval', 1)) == 0) or (epoch + 1 == int(self.arg['num_epoch'])):
                self.eval_epoch(epoch, save_score=False)
                self.print_log(
                    'Best test {}: {:.4f}, epoch: {}'.format(
                        self.best_metric_name,
                        self.best_metric_value if self.best_metric_epoch > 0 else 0.0,
                        self.best_metric_epoch)
                )

        ckpt = torch.load(os.path.join(self.work_dir, 'best.pt'), map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.print_log(f'Final eval with epoch {ckpt["epoch"]} weights')
        self.eval_epoch(epoch=int(ckpt['epoch']) - 1, save_score=True)

        self.print_log('Best test metric ({}): {}'.format(self.best_metric_name, self.best_metric_value))
        self.print_log('Best test metric epoch: {}'.format(self.best_metric_epoch))
        self.print_log('Best test snapshot: {}'.format(self.best_metric_snapshot))


def main():
    parser = argparse.ArgumentParser(description='DeCAPS-Net skeleton branch — ASDPose training')
    parser.add_argument('--config', default='configs/asdpose_skeleton.yaml')
    parser.add_argument('--work-dir', default=None, help='override work_dir from config')
    parser.add_argument('--device', type=int, default=None, help='override device from config')
    parser.add_argument('--phase', default=None, choices=['train', 'test'])
    parser.add_argument('--weights', default=None, help='checkpoint for phase=test')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        arg = yaml.safe_load(f)

    if args.work_dir is not None:
        arg['work_dir'] = args.work_dir
    if args.device is not None:
        arg['device'] = [args.device]
    if args.phase is not None:
        arg['phase'] = args.phase
    if args.weights is not None:
        arg['weights'] = args.weights

    init_seed(int(arg.get('seed', 1)))
    Trainer(arg).start()


if __name__ == '__main__':
    main()
