import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class APRIL():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = model.parameters()

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()

            train_loss = 0.0
            miss_one, miss_two = 0, 0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                total_batches = len(dataloader['train'])
                updates_default = int(getattr(self.args, 'proto_updates_per_epoch', 1))
                effective_updates = updates_default if epochs == 1 else 1

                interval = None
                trigger_steps = set()

                if effective_updates > 1 and total_batches > 0:
                    interval = max(1, total_batches // effective_updates)
                    trigger_steps = set([(i+1) * interval for i in range(effective_updates-1)])
                for batch_idx, batch_data in enumerate(td):

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    if hasattr(self.args, 'mr') and self.args.mr == 0.0:
                        output = model(text, audio, vision, num_modal=3, labels=labels)
                    else:

                        miss_2 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.0]
                        miss_1 = [0.1, 0.2, 0.3, 0.4, 0.3, 0.0, 0.0]
                        denom = (np.round(len(dataloader['train']) / 10) * 10)
                        if self.args.mr > 0 and (0 <= int(self.args.mr * 10 - 1) < len(miss_2)) and\
                            (miss_two / denom < miss_2[int(self.args.mr * 10 - 1)]):
                            output = model(text, audio, vision, num_modal=1, labels=labels)
                            miss_two += 1
                        elif self.args.mr > 0 and (0 <= int(self.args.mr * 10 - 1) < len(miss_1)) and\
                            (miss_one / denom < miss_1[int(self.args.mr * 10 - 1)]):
                            output = model(text, audio, vision, num_modal=2, labels=labels)
                            miss_one += 1
                        else:
                            output = model(text, audio, vision, num_modal=3, labels=labels)

                    combined_loss = self.criterion(output['output_logit'], labels)
                    if hasattr(model, 'prototype_contrastive_loss'):
                        combined_loss = combined_loss + getattr(self.args, 'proto_contrastive_weight', 0.0) * model.prototype_contrastive_loss
                    if hasattr(model, 'personal_recon_loss'):
                        combined_loss = combined_loss + getattr(self.args, 'personal_recon_weight', 0.1) * model.personal_recon_loss

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                    if hasattr(model, 'update_prototypes_epoch_end') and interval is not None:
                        cur_step = batch_idx + 1
                        if cur_step in trigger_steps:
                            model.update_prototypes_epoch_end(epoch=None, vis_callback=None, clear_cache=False)
                if not left_epochs:
                    optimizer.step()

            if hasattr(model, 'update_prototypes_epoch_end'):
                model.update_prototypes_epoch_end(
                    epoch=epochs,
                    vis_callback=None,
                    clear_cache=True
                )

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = Path(self.args['model_save_path'])
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(model_save_path))

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            miss_one, miss_two = 0, 0
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    if hasattr(self.args, 'mr') and self.args.mr == 0.0:
                        output = model(text, audio, vision, num_modal=3, labels=None)
                    else:
                        # 有关怎么模拟缺失的所有内容，都是和前人工作imder对齐。
                        miss_2 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.0]
                        miss_1 = [0.1, 0.2, 0.3, 0.4, 0.3, 0.0, 0.0]
                        denom = (np.round(len(dataloader) / 10) * 10)
                        if self.args.mr > 0 and (0 <= int(self.args.mr * 10 - 1) < len(miss_2)) and\
                            (miss_two / denom < miss_2[int(self.args.mr * 10 - 1)]):
                            output = model(text, audio, vision, num_modal=1, labels=None)
                            miss_two += 1
                        elif self.args.mr > 0 and (0 <= int(self.args.mr * 10 - 1) < len(miss_1)) and\
                            (miss_one / denom < miss_1[int(self.args.mr * 10 - 1)]):
                            output = model(text, audio, vision, num_modal=2, labels=None)
                            miss_one += 1
                        else:
                            output = model(text, audio, vision, num_modal=3, labels=None)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
