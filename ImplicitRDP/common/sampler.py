from typing import Optional
import numpy as np
import numba
from ImplicitRDP.common.replay_buffer import ReplayBuffer
from loguru import logger


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    indices_to_episode_idx = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
            indices_to_episode_idx.append(i)
    indices = np.array(indices)
    return indices, indices_to_episode_idx

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def get_val_mask_one_by_one(n_episodes, val_ratio, seed=0):
    if val_ratio <= 0:
        val_mask = np.zeros(n_episodes, dtype=bool)
        return val_mask

    rng = np.random.default_rng(seed=seed)
    val_mask = rng.random(n_episodes) < val_ratio
    return val_mask

def get_finetune_train_mask(val_mask, finetune_ratio, seed=0):
    n_val = np.sum(val_mask)
    n_episodes = len(val_mask)
    val_ratio = (n_val / n_episodes)
    assert finetune_ratio < val_ratio, f"finetune_ratio {finetune_ratio} should be less than val_ratio {val_ratio}"

    finetune_mask = np.zeros(n_episodes, dtype=bool)
    # have at least 1 episode for finetune validation, and at least 1 episode for finetune train
    val_idxs = np.nonzero(val_mask)[0]
    n_finetune = min(max(1, round(n_episodes * finetune_ratio)), n_val-1)
    rng = np.random.default_rng(seed=seed)
    finetune_idxs = rng.choice(val_idxs, size=n_finetune, replace=False)
    finetune_mask[finetune_idxs] = True
    return finetune_mask

def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        else:
            episode_mask = episode_mask.copy()

        if np.any(episode_mask):
            indices, indices_to_episode_idx = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)
            indices_to_episode_idx = np.zeros((0,), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.episode_mask = episode_mask
        self.indices = indices 
        self.indices_to_episode_idx = indices_to_episode_idx
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        result = dict()
        for key in self.keys:
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
                = self.indices[idx]
            sequence_length = self.sequence_length
            if key not in self.key_first_k:
                first_k = None
            else:
                first_k = self.key_first_k[key]

            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if first_k is None:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(first_k, n_data)
                # fill value with NaN (float) / 0 (int) to catch bugs:
                # the non-loaded region should never be used. NaN can't be cast
                # to integer dtypes — using 0 there avoids a numpy cast warning.
                fill = np.nan if np.issubdtype(input_arr.dtype, np.floating) else 0
                sample = np.full((n_data,) + input_arr.shape[1:],
                    fill_value=fill, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                data = np.zeros(
                    shape=(sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
