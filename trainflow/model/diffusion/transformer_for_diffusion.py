from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from trainflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from trainflow.model.diffusion.transformer_utils import CustomizedTransformerDecoderLayer
from trainflow.model.common.module_attr_mixin import ModuleAttrMixin
import os
import pickle

logger = logging.getLogger(__name__)


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 action_horizon: int,
                 #------------------
                 n_obs_steps: int = None,
                 cond_dim: int = 0,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_emb: int = 768,
                 max_cond_tokens: int=800,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 time_as_cond: bool = True,
                 obs_as_cond: bool = False,
                 n_cond_layers: int = 0,
                 ) -> None:
        super().__init__()

        input_dim = shape_meta['action']['shape'][0]
        output_dim = shape_meta['action']['shape'][0]

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = action_horizon

        T = action_horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_tokens, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            self.decoder_blocks = nn.ModuleList(
                [
                    CustomizedTransformerDecoderLayer(
                        d_model=n_emb,
                        nhead=n_head,
                        dim_feedforward=4 * n_emb,
                        dropout=p_drop_attn,
                        activation='gelu',
                        batch_first=True,
                        norm_first=True  # important for stability
                    )
                    for _ in range(n_layer)
                ]
            )
        else:
            raise NotImplementedError("Encoder only model is not implemented yet.")
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        # constants
        self.T = T
        self.T_cond = T_cond
        self.action_horizon = action_horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.pickle_count = 0

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for attr in dir(module):
                if attr.startswith('weight_'):
                    torch.nn.init.normal_(getattr(module, attr), mean=0.0, std=0.02)
                if attr.startswith('bias_'):
                    torch.nn.init.zeros_(getattr(module, attr))
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif (pn.endswith("weight") or pn.startswith("weight")) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif (pn.endswith("weight") or pn.startswith("weight")) and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        temporal_cond: Optional[torch.Tensor]=None,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        if self.encoder_only:
            raise NotImplementedError("Encoder only model is not implemented yet.")
        
        # 2. process conditions
        # encoder
        cond_emb = time_emb
        if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(cond)
            # (B,To,n_emb)
            cond_emb = torch.cat([cond_emb, cond_obs_emb], dim=1)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[
                                :, :tc, :
                                ]  # each position maps to a (learnable) vector
        cond_emb = self.drop(cond_emb + cond_pos_emb)
        cond_emb = self.encoder(cond_emb)
        # (B,T_cond,n_emb)

        # (optional) 3. process temporal conditions
        temporal_cond_emb = temporal_cond
        if temporal_cond_emb is not None:
            assert temporal_cond_emb.shape[1] == sample.shape[1], "Temporal condition and sample must have the same time dimension"
            ttc = temporal_cond_emb.shape[1]
            temporal_cond_pos_emb = self.cond_pos_emb[
                :, tc:tc+ttc, :
            ]
            temporal_cond_emb = self.drop(temporal_cond_emb + temporal_cond_pos_emb)

        # 4. process input
        # decoder
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        input_emb = self.drop(input_emb + pos_emb)

        # 5. transformer
        # (B,T,n_emb)
        causal_mask = torch.triu(torch.ones((t, t), device=sample.device), diagonal=1).bool()
        tgt = input_emb
        tgt_mask = causal_mask
        if temporal_cond_emb is not None:
            memory = torch.cat([cond_emb, temporal_cond_emb], dim=1)
            # causal mask for temporal condition
            memory_mask = torch.zeros((t, tc+t), device=sample.device).bool()
            memory_mask[:, tc:] = causal_mask
            temporal_memory = None
            temporal_memory_mask = None
        else:
            memory = cond_emb
            memory_mask = None
            temporal_memory = temporal_cond_emb
            temporal_memory_mask = causal_mask
        
        for layer, block in enumerate(self.decoder_blocks):
            tgt = block(tgt,
                        memory,
                        temporal_memory=temporal_memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        temporal_memory_mask=temporal_memory_mask,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=None,
                        temporal_memory_key_padding_mask=None)
            
            if hasattr(block, 'last_attn_weights') and layer == 0 and timestep.item() == 0:
                # (B, T, S)
                attn_weights = block.last_attn_weights
                # Average over batch and action tokens (T)
                # equivalent to averaging over heads (already done), batch, and action tokens
                mean_attn = attn_weights[0].mean(dim=(0,))
                # print(f"Layer 0 average attention weights (shape {mean_attn.shape}):\n{mean_attn}")
                attn_dict = {
                    "time": mean_attn[0],
                    "slow_img_obs": torch.sum(mean_attn[1:1+98]).item(),
                    "slow_tcp_obs": torch.sum(mean_attn[1+98:1+100]).item(),
                    "slow_wrench_obs": torch.sum(mean_attn[1+100:1+102]).item(),
                    "temporal_cond": torch.sum(mean_attn[1+102:]).item()
                }
                os.makedirs('toggle', exist_ok=True)
                with open(f'toggle/attn_weights_{self.pickle_count}.pkl', 'wb') as f:
                    pickle.dump(attn_dict, f)
                self.pickle_count += 1

        x = tgt
        # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x
        
