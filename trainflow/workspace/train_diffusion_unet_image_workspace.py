if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import pickle
from trainflow.workspace.base_workspace import BaseWorkspace
from trainflow.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from trainflow.dataset.base_dataset import BaseImageDataset
from trainflow.common.checkpoint_util import TopKCheckpointManager
from trainflow.common.json_logger import JsonLogger
from trainflow.common.pytorch_util import dict_apply, optimizer_to
from trainflow.model.diffusion.ema_model import EMAModel
from trainflow.model.common.lr_scheduler import get_scheduler
from trainflow.model.common.lr_decay import param_groups_lrd
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state

        if 'timm' in cfg.policy.obs_encoder._target_:
            if cfg.training.layer_decay < 1.0:
                assert not cfg.policy.obs_encoder.use_lora
                assert not cfg.policy.obs_encoder.share_rgb_model
                obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                             shape_meta=cfg.shape_meta,
                                                             weight_decay=cfg.optimizer.encoder_weight_decay,
                                                             no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                             layer_decay=cfg.training.layer_decay)
                count = 0
                for group in obs_encorder_param_groups:
                    count += len(group['params'])
                if cfg.policy.obs_encoder.feature_aggregation == 'map':
                    obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                    for _ in self.model.obs_encoder.attn_pool.parameters():
                        count += 1
                print(f'obs_encorder params: {count}')
                param_groups = [{'params': self.model.model.parameters()}]
                param_groups.extend(obs_encorder_param_groups)
            else:
                obs_encorder_lr = cfg.optimizer.lr
                if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                    obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                    print('==> reduce pretrained obs_encorder\'s lr')
                obs_encorder_params = list()
                for param in self.model.obs_encoder.parameters():
                    if param.requires_grad:
                        obs_encorder_params.append(param)
                print(f'obs_encorder params: {len(obs_encorder_params)}')
                param_groups = [
                    {'params': self.model.model.parameters()},
                    {'params': obs_encorder_params, 'lr': obs_encorder_lr}
                ]
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('_target_')
            if 'encoder_weight_decay' in optimizer_cfg.keys():
                optimizer_cfg.pop('encoder_weight_decay')
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
                **optimizer_cfg
            )
        else:
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('encoder_weight_decay')
            # hack: use larger learning rate for multiple gpus
            accelerator = Accelerator()
            cuda_count = accelerator.num_processes
            print("###########################################")
            print(f"Number of available CUDA devices: {cuda_count}.")
            print(f"Original learning rate: {optimizer_cfg['lr']}")
            # do not scale lr for multiple gpus because we use samller batch size
            # optimizer_cfg['lr'] = optimizer_cfg['lr'] * cuda_count
            print(f"Updated learning rate: {optimizer_cfg['lr']}")
            print("###########################################")
            self.optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        cuda_count = accelerator.num_processes
        
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        print("###########################################")
        print(f"Number of available CUDA devices: {cuda_count}.")
        print(f"Original training batch size: {dataloader_cfg['batch_size']}.")
        print(f"Original validation batch size: {val_dataloader_cfg['batch_size']}.")
        dataloader_cfg['batch_size'] = dataloader_cfg['batch_size'] // cuda_count
        val_dataloader_cfg['batch_size'] = val_dataloader_cfg['batch_size'] // cuda_count
        print(f"Updated training batch size: {dataloader_cfg['batch_size']}")
        print(f"Updated validation batch size: {val_dataloader_cfg['batch_size']}")
        print("###########################################")
        
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        
        # normalizer = dataset.get_normalizer()
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True
            )

        # device transfer
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # early-stopping state. monitor_key must land in step_log at every
        # checkpoint event (see comment on cfg.training.early_stopping in the
        # workspace yaml).
        es_cfg = cfg.training.get('early_stopping', None)
        es_enabled = bool(es_cfg.enabled) if es_cfg is not None else False
        es_mode = es_cfg.mode if es_enabled else 'min'
        es_best = float('inf') if es_mode == 'min' else float('-inf')
        es_stale = 0

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema and accelerator.is_main_process:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = policy(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        extended_obs_dict = batch['extended_obs']
                        gt_action = batch['action']

                        if 'latent' in cfg.name:
                            dataset_obs_temporal_downsample_ratio = cfg.task.dataset.obs_temporal_downsample_ratio
                            result = policy.predict_action(obs_dict,
                                                           extended_obs_dict=extended_obs_dict,
                                                           dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio)
                        else:
                            result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']

                        all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                        if pred_action.shape[-1] in [12, 13, 19]:
                            # ignore virtual target and stiffness
                            all_preds = all_preds[..., :6]
                            all_gt = all_gt[..., :6]

                        mse = torch.nn.functional.mse_loss(all_preds, all_gt)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                    # also log action-space MSE on a val batch — this is the
                    # checkpoint top-k monitor and the early-stopping signal.
                    if cfg.task.dataset.val_ratio > 0 and len(val_dataloader) > 0:
                        with torch.no_grad():
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            extended_obs_dict = batch['extended_obs']
                            gt_action = batch['action']

                            if 'latent' in cfg.name:
                                dataset_obs_temporal_downsample_ratio = cfg.task.dataset.obs_temporal_downsample_ratio
                                result = policy.predict_action(obs_dict,
                                                               extended_obs_dict=extended_obs_dict,
                                                               dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio)
                            else:
                                result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']

                            all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))
                            if pred_action.shape[-1] in [12, 13, 19]:
                                # ignore virtual target and stiffness
                                all_preds = all_preds[..., :6]
                                all_gt = all_gt[..., :6]

                            step_log['val_action_mse_error'] = torch.nn.functional.mse_loss(all_preds, all_gt).item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                accelerator.wait_for_everyone()
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                    # early-stopping check: only fires on checkpoint events so a
                    # break always happens immediately after latest.ckpt + topk are written.
                    if es_enabled:
                        v = step_log.get(es_cfg.monitor_key)
                        if v is None:
                            raise KeyError(
                                f"early_stopping: monitor_key {es_cfg.monitor_key!r} missing from "
                                f"step_log at epoch {self.epoch}. Ensure checkpoint_every is a "
                                f"multiple of sample_every and val_ratio > 0."
                            )
                        improved = (
                            (es_best - v) > es_cfg.min_delta if es_mode == 'min'
                            else (v - es_best) > es_cfg.min_delta
                        )
                        if improved:
                            es_best = v
                            es_stale = 0
                        else:
                            es_stale += 1
                        step_log['es_stale'] = es_stale
                        step_log['es_best'] = es_best
                        if es_stale >= es_cfg.patience:
                            print(
                                f"early stopping at epoch {self.epoch}: "
                                f"no improvement on {es_cfg.monitor_key} for "
                                f"{es_stale} checkpoint events "
                                f"(best={es_best:.6g}, last={v:.6g})."
                            )
                            policy.train()
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            break

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
