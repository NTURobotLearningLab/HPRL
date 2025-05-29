import os
import time
import pickle
import shutil
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from rl.utils import get_vec_normalize, count_parameters
from pretrain.misc_utils import log_record_dict
from pretrain.utils import analyze_z_bz

optim_list = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}

scheduler_list = {
    'steplr'        : lr_scheduler.StepLR,
    'multistep'     : lr_scheduler.MultiStepLR,
    'exponential'   : lr_scheduler.ExponentialLR,
    'cosine'        : lr_scheduler.CosineAnnealingLR,
    'cosine_restart': lr_scheduler.CosineAnnealingWarmRestarts,
    'plateau'       : lr_scheduler.ReduceLROnPlateau,
}

def dfs_freeze(model):
    model.eval()  # disable dropout, batchnorm updates, etc.
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class BaseModel(object):

    def __init__(self, Net, device, config, envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = envs
        self.dsl = dsl

        # build policy network
        self.net = Net(envs, **config)
        self.net.to(device)

        # set number of program tokens
        self.num_program_tokens = self.net.num_program_tokens

        # Load parameters if available
        ckpt_path = config['net']['saved_params_path']
        if ckpt_path is not None:
            try:
                self.load_checkpoint(ckpt_path)
                self.logger.debug('Checkpoint loaded from {}'.format(ckpt_path))
            except:
                self.load_net(ckpt_path)

        # disable some parts of network if don't want to train them
        if config['net']['decoder']['freeze_params']:
            assert config['algorithm'] != 'supervisedRL'
            dfs_freeze(self.net.vae.decoder)

        logger.info('VAE Network:\n{}'.format(self.net.vae))
        logger.info('VAE Network total parameters: {}'.format(count_parameters(self.net.vae)))

        # Initialize optimizer
        self.setup_optimizer(self.net.parameters())

        # Initialize learning rate scheduler
        self.setup_lr_scheduler()

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Initialize epoch number
        self.epoch = 0

        self.program_frozen = False
        self.start_decoder_finetune = False
        self.gen_program_dir = os.path.join(self.config['outdir'], 'generated_programs')
        if not os.path.exists(self.gen_program_dir):
            os.makedirs(self.gen_program_dir)

    # FIXME: implement gradien clipping
    def setup_optimizer(self, parameters):
        cfg = self.config.get('optimizer', {})
        opt_cls = optim_list[cfg['name'].lower()]
        self.optimizer = opt_cls(
            filter(lambda p: p.requires_grad, parameters),
            **cfg.get('params', {})
        )

    def setup_lr_scheduler(self):
        self.scheduler = None
        sched_cfg = self.config['optimizer'].get('scheduler')
        if sched_cfg:
            cfg = sched_cfg.copy()              # don’t mutate original
            sched_name = cfg.pop('name', 'steplr').lower()
            sched_cls  = scheduler_list[sched_name]
            self.scheduler = sched_cls(self.optimizer, **cfg)
            self.logger.debug(f'Using LR scheduler {sched_name}: {cfg}')

    def step_lr_scheduler(self, metric: float = None):
        if not self.scheduler:
            return
        # ReduceLROnPlateau needs the monitored metric; others don’t.
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        try:
            lrs = self.scheduler.get_last_lr()
        except AttributeError:                  # fallback for old Torch
            lrs = [g['lr'] for g in self.optimizer.param_groups]
        self.logger.debug('Learning rate: ' + ', '.join(map(str, lrs)))


    def _add_program_latent_vectors(self, optional_record_dict, optional_record_dict_eval, type='best'):
        self.global_logs['info']['logs']['validation'][type + '_program_latent_vectors'] = optional_record_dict_eval[
            'program_latent_vectors']
        self.global_logs['info']['logs']['validation'][type + '_program_ids'] = optional_record_dict_eval['program_ids']

        self.global_logs['info']['logs']['train'][type + '_program_latent_vectors'] = optional_record_dict[
            'program_latent_vectors']
        self.global_logs['info']['logs']['train'][type+'_program_ids'] = optional_record_dict['program_ids']

    def _run_epoch(self, data_loader, mode, epoch, *args, **kwargs):
        epoch_info = {}
        optinal_epoch_info = {}
        num_batches = len(data_loader)

        batch_info_list = defaultdict(list)
        batch_gt_programs, batch_z_pred_programs, batch_b_z_pred_programs, batch_gen_programs = [], [], [], []
        batch_program_ids, batch_latent_programs = [], []
        batch_latent_behaviors = []
        for batch_idx, batch in enumerate(data_loader):

            batch_info = self._run_batch(batch, mode)

            # log losses and accuracies
            for key, val in batch_info.items():
                if ('loss' in key or 'accuracy' in key):
                    batch_info_list[key].append(val)
                    vtype = 'loss' if 'loss' in key else 'accuracy'
                    if self.writer is not None:
                        self.writer.add_scalar('{}_{}/batch_{}'.format(mode, vtype, key), val,
                                           epoch * num_batches + batch_idx)
                if 'pre_tanh_z' in key:
                    batch_info_list[key].append(val)
                    vtype = 'pre_tanh_z'
                    if self.writer is not None:
                        self.writer.add_scalar('{}_{}/batch_{}'.format(mode, vtype, key), val,
                                           epoch * num_batches + batch_idx)
                if 'time' in key:
                    batch_info_list[key].append(val)
                    vtype = 'time'
                    if self.writer is not None:
                        self.writer.add_scalar('{}_{}/batch_{}'.format(mode, vtype, key), val,
                                           epoch * num_batches + batch_idx)
                if 'zbz' in key:
                    batch_info_list[key].append(val)
                    vtype = 'zbz'
                    if self.writer is not None:
                        self.writer.add_scalar('{}_{}/batch_{}'.format(mode, vtype, key), val,
                                           epoch * num_batches + batch_idx)

            # log programs
            batch_gt_programs.append(batch_info['gt_programs'])
            batch_z_pred_programs.append(batch_info['z_pred_programs'])
            batch_b_z_pred_programs.append(batch_info['b_z_pred_programs'])
            batch_program_ids.append(batch_info['program_ids'])
            batch_latent_programs.append(batch_info['latent_vectors'])
            batch_latent_behaviors.append(batch_info['behavior_vectors'])

            self.logger.debug("epoch:{} batch:{}/{} current batch loss: {}".format(epoch, batch_idx, num_batches,
                                                                                   batch_info['total_loss']))
            if mode == 'eval' and self.writer is not None:
                for i in range(min(batch_info['gt_programs'].shape[0], 5)):
                    self.writer.add_text('dataset/epoch_{}'.format(epoch),
                                         'gt: {} pred: {}'.format(self.dsl.intseq2str(batch_info['gt_programs'][i]),
                                                                  self.dsl.intseq2str(batch_info['z_pred_programs'][i])),
                                         epoch * num_batches)

                batch_gen_programs.append(batch_info['z_generated_programs'])
                for i, program in enumerate(batch_info['z_generated_programs']):
                    self.writer.add_text('generated/epoch_{}'.format(epoch), program, epoch * num_batches)
        
        # Log the averaged evaluation metrics at the end of evaluation
        if mode == 'eval' and hasattr(self, 'eval_metrics') and self.eval_metrics:
            import wandb
            # For accessing global_train_step from the SupervisedModel
            global_step = 0
            if hasattr(self, 'global_train_step'):
                global_step = self.global_train_step
            
            # Average all collected metrics
            avg_metrics = {}
            for key, values in self.eval_metrics.items():
                if values:  # Only average if we have values
                    avg_metrics[f'eval/{key}'] = sum(values) / len(values)
            
            # Log the averaged metrics (all as flat keys)
            wandb.log(avg_metrics, step=global_step)
            program_txt_path = os.path.join(self.gen_program_dir, f'decoded_vs_gt_{epoch}.txt')
            with open(program_txt_path, 'a') as f:
                for i in range(len(batch_info['gt_programs'])):
                    z = torch.tensor(batch_info['latent_vectors'][i])
                    bz = torch.tensor(batch_info['behavior_vectors'][i])
                    gt_str = self.dsl.intseq2str(batch_info['gt_programs'][i])
                    z_pred_str = batch_info['z_generated_programs'][i]
                    b_z_pred_str = batch_info['b_z_generated_programs'][i]
                    f.write(f"truth : {gt_str}\n")
                    f.write(f"z_pred: {z_pred_str}\n")
                    f.write(f"bz_pre: {b_z_pred_str}\n")
                    f.write(f"cosine similarity: {torch.nn.functional.cosine_similarity(z, bz, dim=0).item()}\n")
                    f.write(f"mse: {torch.nn.functional.mse_loss(z, bz).item()}\n")
                    f.write(f"norm_ratio: {torch.norm(z)/torch.norm(bz)}\n\n")
            wandb.save(program_txt_path)
            # Clear the metrics for next evaluation
            self.eval_metrics.clear()

        epoch_info['generated_programs'] = batch_gen_programs
        optinal_epoch_info['program_ids'] = batch_program_ids
        optinal_epoch_info['program_latent_vectors'] = batch_latent_programs
        optinal_epoch_info['behavior_latent_vectors'] = batch_latent_behaviors
        for key, val in batch_info_list.items():
            if ('loss' in key or 'accuracy' in key):
                vtype = 'loss' if 'loss' in key else 'accuracy'
                epoch_info['mean_'+key] = np.mean(np.array(val).flatten())
                if self.writer is not None:
                    self.writer.add_scalar('{}_{}/epoch_{}'.format(mode, vtype, key), epoch_info['mean_'+key], epoch)
            if 'pre_tanh_z' in key:
                vtype = 'pre_tanh_z'
                epoch_info['mean_'+key] = np.mean(np.array(val).flatten())
                if self.writer is not None:
                    self.writer.add_scalar('{}_{}/epoch_{}'.format(mode, vtype, key), epoch_info['mean_'+key], epoch)
            if 'zbz' in key:
                vtype = 'zbz'
                epoch_info['mean_'+key] = np.mean(np.array(val).flatten())
                if self.writer is not None:
                    self.writer.add_scalar('{}_{}/epoch_{}'.format(mode, vtype, key), epoch_info['mean_'+key], epoch)

        return epoch_info, optinal_epoch_info

    def run_one_epoch(self, epoch, best_valid_epoch, best_valid_loss, tr_loader, val_loader, *args, **kwargs):
        self.logger.debug('\n' + 40 * '%' + '    EPOCH {}   '.format(epoch) + 40 * '%')
        self.epoch = epoch

        # Run train epoch
        t = time.time()
        record_dict, optional_record_dict = self._run_epoch(tr_loader, 'train', epoch, *args, **kwargs)

        # log all items in dict
        log_record_dict('train', record_dict, self.global_logs)
        # produce print-out
        if self.verbose:
            self._print_record_dict(record_dict, 'train', time.time() - t)

        if val_loader is not None:
            # Run valid epoch
            t = time.time()
            record_dict_eval, optional_record_dict_eval = self._run_epoch(val_loader, 'eval', epoch,
                                                                          self.config['valid']['debug_samples'],
                                                                          'valid_e{}'.format(epoch),
                                                                          batch_size=self.config['valid']['batch_size'])

            # add logs
            log_record_dict('validation', record_dict_eval, self.global_logs)

            # produce print-out
            if self.verbose:
                self._print_record_dict(record_dict_eval, 'validation', time.time() - t)
            if self.config['freeze_p2p'] and not self.program_frozen and record_dict_eval['mean_z_decoder_greedy_token_accuracy'] > 80:
                self.logger.info(f"Freezing program encoder and decoder at epoch {epoch}")
                dfs_freeze(self.net)
                for param in self.net.vae.behavior_encoder.parameters():
                    param.requires_grad = True
                self.net.vae.behavior_encoder.train()
                self.program_frozen = True
                # Also re-initialize the optimizer to exclude frozen parameters
                self.setup_optimizer(self.net.parameters())
            if self.config['stop_teacher_enforcing'] and self.net.teacher_enforcing and record_dict_eval['mean_z_decoder_greedy_token_accuracy'] > 10:
                self.logger.info(f"Stop teacher enforcing at epoch {epoch}")
                self.net.teacher_enforcing = False
            
            # print(f"record mean andgle: {record_dict_eval['mean_zbz_angle_deg']}")
            # print(f"record mean scale ratio: {record_dict_eval['mean_zbz_scale_ratio']}")
            if self.config['finetune_decoder'] and not self.start_decoder_finetune and record_dict_eval['mean_z_decoder_greedy_token_accuracy'] > 80\
                and record_dict_eval['mean_zbz_angle_deg'] < 15 and record_dict_eval['mean_zbz_scale_ratio'] < 1.1 and record_dict_eval['mean_zbz_scale_ratio'] > 0.9:
            # if self.config['finetune_decoder'] and not self.start_decoder_finetune:    
                self.logger.info(f"Finetuning decoder at epoch {epoch}")
                # print("Finetuning decoder at epoch {}".format(epoch))
                dfs_freeze(self.net)
                for param in self.net.vae.decoder.parameters():
                    param.requires_grad = True
                self.net.vae.decoder.train()
                self.start_decoder_finetune = True
                # Also re-initialize the optimizer to exclude frozen parameters
                self.setup_optimizer(self.net.parameters())

            if record_dict_eval['mean_total_loss'] < best_valid_loss:
                best_valid_epoch = epoch
                best_valid_loss = record_dict_eval['mean_total_loss']
                self._add_program_latent_vectors(optional_record_dict, optional_record_dict_eval, type='best')
                try:
                    self.save_checkpoint(
                        os.path.abspath(os.path.join(self.config['outdir'], 'best_valid_params.ptp'.format(epoch))))
                    self.logger.info(f"saved checkpoints to {os.path.abspath(os.path.join(self.config['outdir'], 'best_valid_params.ptp'.format(epoch)))}")
                except:
                    self.save_net(
                        os.path.abspath(os.path.join(self.config['outdir'], 'best_valid_params.ptp'.format(epoch))))
                

            if np.isnan(record_dict_eval['mean_total_loss']):
                self.logger.debug(self.verbose, 'Early Stopping because validation loss is nan')
                return best_valid_epoch, best_valid_loss, True

        # Perform LR scheduler step
        self.step_lr_scheduler()

        # Save net
        self._add_program_latent_vectors(optional_record_dict, optional_record_dict_eval, type='final')
        try:
            self.save_checkpoint(os.path.join(self.config['outdir'], 'final_params.ptp'))
            self.logger.info(f"saved checkpoints to {os.path.join(self.config['outdir'], 'final_params.ptp')}")
        except:
            self.save_net(os.path.join(self.config['outdir'], 'final_params.ptp'))

        # Save results
        pickle.dump(self.global_logs, file=open(os.path.join(self.config['outdir'], self.config['record_file']), 'wb'))

        return best_valid_epoch, best_valid_loss, record_dict_eval, False

    def train(self,  train_dataloader, val_dataloader, *args, **kwargs):
        tr_loader = train_dataloader
        val_loader = val_dataloader

        # Initialize params
        max_epoch = kwargs['max_epoch']

        # Train epochs
        best_valid_loss = np.inf
        best_valid_epoch = 0

        for epoch in range(max_epoch):
            best_valid_epoch, best_valid_loss, record_dict_eval, done = self.run_one_epoch(epoch, best_valid_epoch,
                                                                                           best_valid_loss, tr_loader,
                                                                                           val_loader, *args, **kwargs)
            assert not done, 'found NaN in parameters'

        return None

    def evaluate(self, data_loader, epoch=0, *args, **kwargs):
        t = time.time()

        if self.config['mode'] == 'eval':
            assert self.config['net'][
                       'saved_params_path'] is not None, 'need trained parameters to evaluate, got {}'.format(
                self.config['net']['saved_params_path'])

        epoch_records, optinal_epoch_records = self._run_epoch(data_loader, 'eval', epoch, *args, **kwargs)
        epoch_z = optinal_epoch_records['program_latent_vectors']
        epoch_bz = optinal_epoch_records['behavior_latent_vectors']
        analyze_z_bz(epoch_z, epoch_bz)
        # Log and print epoch records
        log_record_dict('eval', epoch_records, self.global_logs)
        self._print_record_dict(epoch_records, 'Eval', time.time() - t)
        self.global_logs['result'].update({
            'loss': epoch_records['mean_total_loss'],
        })

        # Save results
        pickle.dump(self.global_logs, file=open(
            os.path.join(self.config['outdir'], self.config['record_file'].replace('.pkl', '_eval.pkl')), 'wb'))

# --- add these imports near the top -------------
    from typing import Optional, Dict
    def save_checkpoint(
        self,
        filename: str,
        extra_payload: Optional[Dict] = None,
    ):
        """
        Stores model, optimizer, scheduler and VecNormalize statistics.
        """
        checkpoint = {
            "net_state_dict": self.net.state_dict(),
            "ob_rms": getattr(get_vec_normalize(self.envs), "ob_rms", None),
            "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None else None,
            "epoch": self.epoch,
        }
        if extra_payload:
            checkpoint.update(extra_payload)

        torch.save(checkpoint, filename)
        self.logger.debug(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename: str, strict: bool = False):
        """
        Restores model, optimizer and scheduler states (if they exist in the file).
        Call *after* optimizer / scheduler have been created.
        """
        self.logger.debug(f"Loading checkpoint from {filename}")
        ckpt = torch.load(filename, map_location=self.device)

        # — network weights —
        self.net.load_state_dict(ckpt["net_state_dict"], strict=strict)

        # — vec-normalize running stats —
        if ckpt.get("ob_rms") is not None:
            vec_norm = get_vec_normalize(self.envs)
            if vec_norm:
                vec_norm.ob_rms = ckpt["ob_rms"]

        # — optimizer —
        if self.optimizer is not None and ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # — scheduler —
        if self.scheduler is not None and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # — epoch counter (optional) —
        self.epoch = ckpt.get("epoch", 0)
    def save_net(self, filename):
        params = [self.net.state_dict(), getattr(get_vec_normalize(self.envs), 'ob_rms', None)]
        torch.save(params, filename)
        self.logger.debug('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.debug('Loading params from {}'.format(filename))
        params = torch.load(filename, map_location=self.device)
        self.net.load_state_dict(params[0], strict=False)

    def _print_record_dict(self, record_dict, usage, t_taken):
        loss_str = ''
        for k, v in record_dict.items():
            if 'loss' not in k and 'accuracy' not in k:
                continue
            loss_str = loss_str + ' {}: {:.4f}'.format(k, v)

        self.logger.debug('{}:{} took {:.3f}s'.format(usage, loss_str, t_taken))
        return None
