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
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import copy
import random
import tqdm
import numpy as np
import pickle

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.checkpoint_util import TopKCheckpointManager
from ImplicitRDP.common.json_logger import JsonLogger
from ImplicitRDP.model.diffusion.ema_model import EMAModel
from ImplicitRDP.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

from loguru import logger

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionTransformerImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerImagePolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
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
        self.optimizer = self.model.get_optimizer(**optimizer_cfg)

        self.global_step = 0
        self.epoch = 0
        
        # Initialize AMP scaler if AMP is enabled
        self.scaler = None
        if cfg.training.get('use_amp', False):
            self.scaler = GradScaler()
        
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer', 'scaler']

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # Get AMP configuration
        use_amp = cfg.training.get('use_amp', False)
        amp_dtype = cfg.training.get('amp_dtype', 'bf16')
        
        # Determine autocast dtype
        if use_amp:
            if amp_dtype == 'bf16' and torch.cuda.is_bf16_supported():
                logger.info(f"Using bf16 for AMP")
                amp_dtype = torch.bfloat16
            elif amp_dtype == 'fp16':
                logger.info(f"Using fp16 for AMP")
                amp_dtype = torch.float16
            else:
                logger.warning(f"AMP dtype {amp_dtype} not supported or bf16 not available, falling back to fp16")
                amp_dtype = torch.float16
        else:
            amp_dtype = None
        
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
            assert cfg.training.pretrained_ckpt_path is None, "pretrained_ckpt_path and resume cannot be used together"
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # init model from pretrained checkpoint
        if cfg.training.pretrained_ckpt_path is not None:
            ckpt_path = cfg.training.pretrained_ckpt_path
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            # handle conda_pos_emb
            cond_pos_emb = payload['state_dicts']['model']['model.cond_pos_emb']
            ema_cond_pos_emb = payload['state_dicts']['ema_model']['model.cond_pos_emb']
            with torch.no_grad():
                self.model.model.cond_pos_emb[:, :cond_pos_emb.shape[1], :] = cond_pos_emb
                self.ema_model.model.cond_pos_emb[:, :ema_cond_pos_emb.shape[1], :] = ema_cond_pos_emb
            del payload['state_dicts']['model']['model.cond_pos_emb']
            del payload['state_dicts']['ema_model']['model.cond_pos_emb']
            # handle normalizer
            for key in self.model.state_dict():
                if 'normalizer' in key:
                    payload['state_dicts']['model'][key] = self.model.state_dict()[key]
                    payload['state_dicts']['ema_model'][key] = self.model.state_dict()[key]
            self.load_payload(payload, exclude_keys=['optimizer', 'scaler'], include_keys=[], strict=False)
            logger.info(f"Loaded model from checkpoint: {ckpt_path}")

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

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            if cfg.training.pretrained_ckpt_path is not None:
                normalizer_state_dict = normalizer.state_dict()
                current_normalizer_state_dict = self.model.normalizer.state_dict()
                for key in current_normalizer_state_dict.keys():
                    normalizer_state_dict[key] = current_normalizer_state_dict[key]
                normalizer.load_state_dict(normalizer_state_dict)
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

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

        # # configure logging
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

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

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

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

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

                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            raw_loss = self.model.compute_diffusion_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            
                        # Scale loss and backward pass
                        if use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                            
                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
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
                train_loss = np.nanmean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                with autocast(enabled=use_amp, dtype=amp_dtype):
                                    loss = policy.compute_diffusion_loss(batch)
                                    val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)[~torch.isnan(torch.tensor(val_losses))]).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1)
                    gt_action = gt_action.view(B, T, -1)
                    step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        batch_names = ['train']
                        samping_batchs = [train_sampling_batch]
                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch_names.append('val')
                            samping_batchs.append(val_sampling_batch)
                        for batch_name, sampling_batch in zip(batch_names, samping_batchs):
                            # evaluate difference
                            batch = dict_apply(sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            extended_obs_dict = batch['extended_obs']
                            gt_action = batch['action']

                            if 'reactive' in cfg.name:
                                dataset_obs_temporal_downsample_ratio = cfg.task.dataset.obs_temporal_downsample_ratio
                                result = policy.predict_action(obs_dict,
                                                            dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
                                                            extended_obs_dict=extended_obs_dict)
                            else:
                                result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']

                            log_action_mse(step_log, batch_name, pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action
                
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
                # ========= eval end for this epoch ==========
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
    workspace = TrainDiffusionTransformerImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
