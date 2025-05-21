# text2music_dataset.py

import torch
import numpy as np
import random
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets, Dataset as HFDataset # Added concatenate_datasets
from loguru import logger
import traceback
import warnings
import os
import torch.nn.functional as F
from tqdm import tqdm


warnings.simplefilter("ignore", category=FutureWarning)

# DEFAULT_TRAIN_PATH should now point to the cached dataset location by default
# This can be overridden by the trainer's `dataset_path` hparam.
DEFAULT_TRAIN_PATH = "cache"

class Text2MusicDataset(Dataset):
    def __init__(self, train=True, train_dataset_path=DEFAULT_TRAIN_PATH,
                 sample_size=None, shuffle_flag=True):
        """
        Initialize the Text2Music dataset from pre-processed cache with validation.

        Args:
            train: Boolean, if true, indicates training mode (can affect shuffling).
            train_dataset_path: Path to the cached Hugging Face dataset.
            sample_size: Optional limit on number of samples to use.
            shuffle_flag: Whether to shuffle the dataset upon loading (for training).
        """
        self.train_dataset_path = train_dataset_path
        
        logger.info(f"Loading cached dataset from: {self.train_dataset_path}")
        try:
            self.dataset = load_from_disk(self.train_dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.train_dataset_path}. "
                        f"Ensure preprocess.py has run successfully and the path is correct. Error: {e}")
            raise

        # Define required keys and expected types
        required_keys_types = {
            "target_latents": (np.ndarray, list),
            "encoder_text_hidden_states": (np.ndarray, list),
            "text_attention_mask": (np.ndarray, list),
            "speaker_embds": (np.ndarray, list),
            "lyric_token_ids": (np.ndarray, list),
            "lyric_masks": (np.ndarray, list),
            "mert_ssl_hidden_states": (np.ndarray, list),
            "mhubert_ssl_hidden_states": (np.ndarray, list),
            "keys": str,
            "processed_prompt_text": str,
            "latent_seq_len": int,
            "processed_wav_length": (int, np.int64),
        }

        #just return 1 irght now
        # self.total_samples = 1
        # self.dataset = self.dataset.select(range(self.total_samples))
        # return

        # # Validate dataset
        # logger.info("Validating dataset contents...")
        # invalid_items = []
        # for idx, item in enumerate(self.dataset):
        #     #right now only do 1
        #     if idx > 1:
        #         break
        #     item_key = item.get("keys", f"Unknown_at_idx_{idx}")
        #     missing_keys = [key for key, expected_type in required_keys_types.items()
        #                     if key not in item]
        #     if missing_keys:
        #         invalid_items.append((idx, item_key, f"Missing keys: {missing_keys}"))
        #         continue
            
        #     # Check types
        #     type_mismatches = []
        #     for key, expected_type in required_keys_types.items():
        #         if not isinstance(item[key], expected_type):
        #             type_mismatches.append(f"{key}: expected {expected_type}, got {type(item[key])}")
        #     if type_mismatches:
        #         invalid_items.append((idx, item_key, f"Type mismatches: {type_mismatches}"))

        # if invalid_items:
        #     for idx, key, issue in invalid_items:
        #         logger.warning(f"Invalid item at index {idx} (key: {key}): {issue}")
        #     # Option 1: Filter invalid items
        #     valid_indices = [i for i in range(len(self.dataset)) if i not in [x[0] for x in invalid_items]]
        #     original_size = len(self.dataset)
        #     self.dataset = self.dataset.select(valid_indices)
        #     logger.info(f"Filtered dataset: {original_size} -> {len(self.dataset)} samples after removing {len(invalid_items)} invalid items.")
        #     if len(self.dataset) == 0:
        #         logger.error("No valid items remain after filtering!")
        #         raise ValueError("Dataset is empty after validation filtering.")
        # else:
        #     logger.info("Dataset validation passed: all items contain required keys with correct types.")

        # # Apply sample_size if specified
        # if sample_size is not None and sample_size > 0 and sample_size < len(self.dataset):
        #     logger.info(f"Using a subset of {sample_size} samples from the dataset.")
        #     self.dataset = self.dataset.select(range(sample_size))
        
        # # Shuffle if requested
        # if shuffle_flag and train:
        #     logger.info("Shuffling pre-processed dataset.")
        #     self.dataset = self.dataset.shuffle(seed=random.randint(0, 2**32 - 1))

        self.total_samples = len(self.dataset)

        if self.total_samples == 0:
            logger.warning(f"Loaded dataset from {self.train_dataset_path} but it contains 0 samples!")
        else:
            logger.info(f"Initialized Text2MusicDataset with {self.total_samples} pre-processed samples.")
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Helper function to convert list to NumPy array if necessary
            def to_numpy(data, expected_dtype):
                if isinstance(data, list):
                    try:
                        return np.array(data, dtype=expected_dtype)
                    except ValueError as e:
                        raise ValueError(f"Failed to convert list to NumPy array for index {idx}: {e}")
                elif isinstance(data, np.ndarray):
                    return data.astype(expected_dtype)
                else:
                    raise TypeError(f"Unexpected type {type(data)} for data at index {idx}")
            
            # Convert fields to NumPy arrays with appropriate dtypes
            target_latents = to_numpy(item["target_latents"], np.float32)
            encoder_text_hidden_states = to_numpy(item["encoder_text_hidden_states"], np.float32)
            text_attention_mask = to_numpy(item["text_attention_mask"], np.int64)
            speaker_embds = to_numpy(item["speaker_embds"], np.float32)
            lyric_token_ids = to_numpy(item["lyric_token_ids"], np.int64)
            lyric_masks = to_numpy(item["lyric_masks"], np.int64)
            mert_ssl_hidden_states = to_numpy(item["mert_ssl_hidden_states"], np.float32)
            mhubert_ssl_hidden_states = to_numpy(item["mhubert_ssl_hidden_states"], np.float32)
            
            # Convert to PyTorch tensors
            target_latents = torch.from_numpy(target_latents)
            encoder_text_hidden_states = torch.from_numpy(encoder_text_hidden_states)
            text_attention_mask = torch.from_numpy(text_attention_mask)
            speaker_embds = torch.from_numpy(speaker_embds)
            lyric_token_ids = torch.from_numpy(lyric_token_ids)
            lyric_masks = torch.from_numpy(lyric_masks)
            
            # Handle empty SSL hidden states
            if mert_ssl_hidden_states.size == 0:
                mert_ssl_hidden_states = torch.empty(0, 768, dtype=torch.float32)
            else:
                mert_ssl_hidden_states = torch.from_numpy(mert_ssl_hidden_states)
            
            if mhubert_ssl_hidden_states.size == 0:
                mhubert_ssl_hidden_states = torch.empty(0, 1024, dtype=torch.float32)
            else:
                mhubert_ssl_hidden_states = torch.from_numpy(mhubert_ssl_hidden_states)
            
            # Other fields
            keys = item["keys"]
            processed_prompt_text = item["processed_prompt_text"]
            processed_wav_length = torch.tensor(item["processed_wav_length"], dtype=torch.long)
            latent_attention_mask = torch.ones(item["latent_seq_len"], dtype=torch.float32)
            
            # Reconstruct candidate_lyric_chunks
            raw_lyrics = item.get("norm_lyrics", "[instrumental]")
            lyrics_lines = raw_lyrics.split("\n")

            #debug print the lyrics here
            candidate_lyric_chunks = [{"lyric": line} for line in lyrics_lines]
            
            return {
                "keys": keys,
                "target_latents": target_latents,
                "latent_attention_mask": latent_attention_mask,
                "encoder_text_hidden_states": encoder_text_hidden_states,
                "text_attention_mask": text_attention_mask,
                "speaker_embds": speaker_embds,
                "lyric_token_ids": lyric_token_ids,
                "lyric_masks": lyric_masks,
                "mert_ssl_hidden_states_item": mert_ssl_hidden_states,
                "mhubert_ssl_hidden_states_item": mhubert_ssl_hidden_states,
                "wav_lengths": processed_wav_length,
                "prompts": processed_prompt_text,
                "candidate_lyric_chunks": candidate_lyric_chunks
            }
        except Exception as e:
            item_key_info = item.get('keys', 'UNKNOWN_KEY') if isinstance(item, dict) else f'UNKNOWN_ITEM_TYPE ({type(item)})'
            logger.error(f"Error processing cached item at index {idx} (key: {item_key_info}): {e}", exc_info=True)
            # Fallback to a random item to prevent training crash
            if not hasattr(self, '_retrying_idx') or self._retrying_idx != idx:
                self._retrying_idx = idx
                logger.warning(f"Attempting to load a different random item instead of index {idx}.")
                new_idx = random.choice(list(range(idx)) + list(range(idx + 1, len(self)))) if len(self) > 1 else 0
                return self.__getitem__(new_idx)
            else:
                logger.error(f"Repeated error on index {idx} or dataset too small to retry. Raising.")
                raise
    def collate_fn(self, batch_list):
        # Filter out None items that might result from __getitem__ errors
        batch_list = [item for item in batch_list if item is not None]
        if not batch_list:
            # Return None or an empty dictionary if the whole batch is problematic
            # The DataLoader will skip this batch if its collate_fn returns None (depends on DataLoader version/config)
            # Or raise an error:
            # raise ValueError("Batch list is empty after filtering None items.")
            logger.warning("Collate_fn received an empty batch_list after filtering. Returning empty dict.")
            return {}
        elem = batch_list[0]
        collated = {}
        
        # Handle keys that are lists of strings or list of list of dicts
        string_list_keys = ["keys", "prompts"]
        list_of_dicts_keys = ["candidate_lyric_chunks"]

        for key in elem.keys():
            if key in string_list_keys:
                collated[key] = [d[key] for d in batch_list]
            elif key in list_of_dicts_keys:
                collated[key] = [d[key] for d in batch_list] # list of list of dicts
            elif key == "mert_ssl_hidden_states_item":
                collated["mert_ssl_hidden_states"] = [d[key] for d in batch_list] # List of Tensors
            elif key == "mhubert_ssl_hidden_states_item":
                collated["mhubert_ssl_hidden_states"] = [d[key] for d in batch_list] # List of Tensors
            elif isinstance(elem[key], torch.Tensor):
                sequences = [d[key] for d in batch_list]
                if sequences[0].ndim == 0: # Scalar tensors like wav_lengths
                    collated[key] = torch.stack(sequences)
                elif sequences[0].ndim == 1: # e.g., latent_attention_mask, text_attention_mask, lyric_token_ids, lyric_masks
                    collated[key] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
                elif sequences[0].ndim == 2: # e.g., speaker_embds [D], encoder_text_hidden_states [L,D], SSL features [L,D]
                    # speaker_embds are fixed dim, just stack. Others need padding.
                    if key == "speaker_embds":
                        collated[key] = torch.stack(sequences)
                    else: # encoder_text_hidden_states, (old batched SSL features, no longer used)
                        collated[key] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
                elif sequences[0].ndim == 3: # e.g., target_latents [C, H_feat, W_seq]
                    # Pad along W_seq (dim 2)
                    max_len_dim2 = max(s.shape[2] for s in sequences)
                    padded_sequences = []
                    for seq in sequences:
                        pad_len_dim2 = max_len_dim2 - seq.shape[2]
                        padding = (0, pad_len_dim2) 
                        padded_sequences.append(F.pad(seq, padding, "constant", 0))
                    collated[key] = torch.stack(padded_sequences)
                else:
                    logger.error(f"Unsupported tensor ndim for key {key}: {sequences[0].ndim}")
                    raise ValueError(f"Unsupported tensor ndim for key {key}: {sequences[0].ndim}")
            else:
                logger.warning(f"Key {key} with unhandled type {type(elem[key])} in collate_fn.")
                collated[key] = [d[key] for d in batch_list] # Default to list aggregation
        
        return collated

if __name__ == "__main__":
    logger.info("Testing Text2MusicDataset with cached data...")
    # Ensure your DEFAULT_TRAIN_PATH ('./cache' or as configured) exists and has preprocessed data
    if not os.path.exists(DEFAULT_TRAIN_PATH) or not os.listdir(DEFAULT_TRAIN_PATH):
        logger.error(f"Cached dataset not found or empty at {DEFAULT_TRAIN_PATH}. "
                    f"Please run preprocess.py first or provide correct --dataset_path.")
    else:
        try:
            dataset = Text2MusicDataset(train_dataset_path=DEFAULT_TRAIN_PATH, shuffle_flag=False) # No shuffle for deterministic test
            if len(dataset) > 0:
                logger.info(f"Successfully loaded dataset. Number of items: {len(dataset)}")

            item = dataset[0]
            logger.info("\n--- First Item from __getitem__ ---")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}")
                else:
                    logger.info(f"Key: {k}, Type: {type(v)}, Value: {str(v)[:100]}") # Print snippet of string/list

            if len(dataset) > 1:
                item2 = dataset[1]
                batch_list_for_collate = [item, item2]
            else:
                batch_list_for_collate = [item]
            
            logger.info("\n--- Collated Batch (size {}) ---".format(len(batch_list_for_collate)))
            collated_batch = dataset.collate_fn(batch_list_for_collate)
            if collated_batch: # Check if collate_fn returned something
                for k, v in collated_batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}")
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor): # For SSL lists
                        logger.info(f"Key: {k} (List of Tensors), Num Tensors: {len(v)}, First Tensor Shape: {v[0].shape if v else 'N/A'}, Dtype: {v[0].dtype if v else 'N/A'}")
                    else:
                        logger.info(f"Key: {k}, Type: {type(v)}, Value: {str(v)[:100]}")
            else:
                logger.warning("Collate function returned an empty or None batch.")

        except Exception as e:
            logger.error(f"Error during Text2MusicDataset test: {e}", exc_info=True)


