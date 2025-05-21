# trainer.py

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import matplotlib
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from text2music_dataset import Text2MusicDataset
from loguru import logger
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os
from pathlib import Path
from pipeline_ace_step import ACEStepPipeline

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        train: bool = True, # This hparam now mainly controls CFG and model.train()
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0, # Coefficient for SSL projection losses
        checkpoint_dir=None, # For ACEStepPipeline and LoRA
        max_steps: int = 200000,
        warmup_steps: int = 4000,
        dataset_path: str = "./cache", # MODIFIED: Default path to cached dataset
        lora_config_path: str = None,

        new_lora_config_path: str | None = None,
        new_lora_name: str = "lora2",
        freeze_prev_loras: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args
        self.is_train = train # Retain for controlling mode, e.g. CFG, model.train()
        self.T = T # For scheduler
        
        self.scheduler = self.get_scheduler()
        
        # if torch.distributed.is_initialized():
        #     self.local_rank = torch.distributed.get_rank()
        # else:
        #     self.local_rank = 0 # Default for non-distributed

        # Load core transformer and DCAE (for decoding in plot_step)
        logger.info(f"Initializing ACEStepPipeline from checkpoint_dir: {self.hparams.checkpoint_dir}")
        acestep_pipeline = ACEStepPipeline(self.hparams.checkpoint_dir)
        acestep_pipeline.load_checkpoint(self.hparams.checkpoint_dir)

        # Main transformer model to be trained
        self.transformers = acestep_pipeline.ace_step_transformer.float() # Keep on CPU initially, PL handles device
        if self.hparams.lora_config_path is not None:
            logger.info(f"Applying LoRA config from: {self.hparams.lora_config_path}")
            try:
                from peft import LoraConfig
            except ImportError:
                raise ImportError("Please install peft library to use LoRA training: pip install peft")
            with open(self.hparams.lora_config_path, encoding="utf-8") as f:
                lora_config_dict = json.load(f)
            lora_config = LoraConfig(**lora_config_dict)
            self.transformers.add_adapter(adapter_config=lora_config)
            active_adapters_list = []
            if hasattr(self.transformers, 'active_adapters') and callable(self.transformers.active_adapters):
                try:
                    active_adapters_list = self.transformers.active_adapters() # Call the method
                    logger.info(f"Called active_adapters(), result: {active_adapters_list}")
                    if not active_adapters_list:
                        logger.warning("PEFT active_adapters() returned an empty list. The adapter might not be truly active.")
                    # Assuming 'default' is the adapter name PEFT assigns if not specified
                    elif 'default' not in active_adapters_list and active_adapters_list:
                        logger.warning(f"The 'default' adapter is not in the active_adapters list: {active_adapters_list}. This could be the issue.")
                    else:
                        logger.info(f"Active PEFT adapters list seems OK: {active_adapters_list}")
                except Exception as e:
                    logger.error(f"Error calling self.transformers.active_adapters(): {e}")
            elif hasattr(self.transformers, 'active_adapter'): # Fallback for some PEFT model types
                active_adapter_prop = self.transformers.active_adapter
                logger.info(f"Value of active_adapter property: {active_adapter_prop}")
                if not active_adapter_prop:
                    logger.warning("PEFT active_adapter property is None or empty. The adapter might not be truly active.")
            else:
                logger.warning("Could not reliably determine active PEFT adapters from self.transformers.")

            logger.info("--- Transformer structure after adding LoRA adapter ---") # Your existing log
            logger.info(self.transformers)
            logger.info("--- End of transformer structure ---") # Your existing log
            # Check if any LoRA parameters are indeed trainable:
            lora_params_check = [n for n, p in self.transformers.named_parameters() if 'lora_' in n and p.requires_grad]
            logger.info(f"Found {len(lora_params_check)} parameters with 'lora_' in name and requires_grad=True.")
            if not lora_params_check:
                logger.warning("No parameters with 'lora_' in their name found or they don't require grad. Check LoRA config and application.")

        # ─── OPTIONAL SECOND‑STAGE LORA TRAINING ──────────────────────────────────
        # ─── OPTIONAL SECOND‑STAGE LoRA TRAINING ──────────────────────────────────────
        if self.hparams.new_lora_config_path is not None:
            # 2‑a)  Optionally freeze *all* parameters that belong to the
            #       LoRA(s) stored in the checkpoint we just loaded.
            if self.hparams.freeze_prev_loras:
                for n, p in self.transformers.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = False
                logger.info("✅  All existing LoRA params frozen.")

            # 2‑b)  Create and register the *new* adapter
            from peft import LoraConfig
            with open(self.hparams.new_lora_config_path, "r", encoding="utf-8") as f:
                new_cfg = LoraConfig(**json.load(f))

            new_adapter_name = self.hparams.new_lora_name
            self.transformers.add_adapter(adapter_config=new_cfg,
                                        adapter_name=new_adapter_name)

            # 2‑c)  Activate **only** the new adapter for forward & training
            #       (older adapters stay frozen *and* inactive ⇒ no grads / no compute)
            self.transformers.set_adapter(new_adapter_name)
            logger.info(f"🆕  Added trainable adapter “{new_adapter_name}”; it is now the active adapter.")

# ──────────────────────────────────────────────────────────────────────────────




        # DCAE needed for plot_step (decoding latents to audio)
        self.dcae = acestep_pipeline.music_dcae.float() # Keep on CPU initially
        self.dcae.requires_grad_(False)
        
        # SSL models and text encoder are no longer loaded here as their outputs are cached.
        # self.text_encoder_model, self.text_tokenizer = None, None
        # self.mert_model, self.resampler_mert, self.processor_mert = None, None, None
        # self.hubert_model, self.resampler_mhubert, self.processor_mhubert = None, None, None

        if self.is_train:
            self.transformers.train()
        else:
            self.transformers.eval()
        
        self.dcae.eval() # DCAE is always in eval mode for decoding

        # ssl_coeff is still an hparam, as proj_losses come from the transformer model itself
        # These projection losses are assumed to be calculated by self.transformers
        # based on the cached SSL features it receives.

    def load_state_dict(self, state_dict, strict: bool = True):
            """
            We resume from a checkpoint that has no weights for the brand‑new
            LoRA adapter (“lofi”), so we deliberately ignore the missing keys.
            """
            # always load with strict=False
            missing, unexpected = super().load_state_dict(state_dict, strict=False)

            # optional: remember what was skipped so you can print / debug later
            if missing:
                logger.warning(
                    f"[LoRA‑resume]  skipped {len(missing)} missing keys "
                    f"(new adapter or other freshly‑added params).")
            if unexpected:
                logger.warning(
                    f"[LoRA‑resume]  ignored {len(unexpected)} unexpected keys.")

            return missing, unexpected

    # REMOVED: infer_mert_ssl, infer_mhubert_ssl, get_text_embeddings
    # These are now part of the offline preprocessing.

    def preprocess(self, batch, train=True):
        # The batch now comes directly from the (modified) Text2MusicDataset and collate_fn
        # It contains pre-computed tensors.
        # This method is now primarily for applying Classifier-Free Guidance (CFG) masks.

        # Ensure all tensors are on the correct device. PyTorch Lightning should handle this for the batch.
        # Example: target_latents = batch["target_latents"].to(self.device) (if not already done by PL)
        
        keys = batch["keys"]
        target_latents = batch["target_latents"]
        # latent_attention_mask was for target_latents sequence length
        # In run_step, attention_mask is used for the *transformer input* (noisy_image),
        # which has the same shape as target_latents.
        attention_mask = batch["latent_attention_mask"] # This should be padded like target_latents

        encoder_text_hidden_states = batch["encoder_text_hidden_states"]
        text_attention_mask = batch["text_attention_mask"]
        speaker_embds = batch["speaker_embds"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"]
        
        # SSL features are now lists of tensors from the collate_fn
        mert_ssl_hidden_states_list = batch.get("mert_ssl_hidden_states", []) # Default to empty list if key missing
        mhubert_ssl_hidden_states_list = batch.get("mhubert_ssl_hidden_states", [])

        bs = target_latents.shape[0]
        device = target_latents.device # Get device from data
        dtype = target_latents.dtype

        # Ensure consistent device and dtype for all inputs to the model
        # This might be redundant if DataLoader + PL already handle it well.
        # target_latents = target_latents.to(device=device, dtype=dtype)
        # attention_mask = attention_mask.to(device=device, dtype=dtype)
        # encoder_text_hidden_states = encoder_text_hidden_states.to(device=device, dtype=dtype)
        # text_attention_mask = text_attention_mask.to(device=device, dtype=torch.bool if text_attention_mask.dtype == torch.uint8 else text_attention_mask.dtype) # Ensure bool if needed
        # speaker_embds = speaker_embds.to(device=device, dtype=dtype)
        # lyric_token_ids = lyric_token_ids.to(device=device, dtype=torch.long)
        # lyric_mask = lyric_mask.to(device=device, dtype=torch.bool if lyric_mask.dtype == torch.uint8 else lyric_mask.dtype)

        # Reformat SSL hidden states:
        # The transformer expects `ssl_hidden_states` as a list of lists of tensors,
        # e.g., [[mert_b0, mert_b1,...], [mhubert_b0, mhubert_b1,...]]
        # Our current batch["mert_ssl_hidden_states"] is already a list of tensors.
        # So, we just need to wrap these lists.
        
        # This part depends on how self.transformers expects ssl_hidden_states.
        # The original run_step did:
        # all_ssl_hiden_states = []
        # if mert_ssl_hidden_states is not None: all_ssl_hiden_states.append(mert_ssl_hidden_states)
        # if mhubert_ssl_hidden_states is not None: all_ssl_hiden_states.append(mhubert_ssl_hidden_states)
        # Here, mert_ssl_hidden_states was a list of tensors (one per batch item).
        # So, all_ssl_hiden_states became a list of [list_of_mert_tensors, list_of_mhubert_tensors].
        # The Text2MusicDataset now provides these lists directly in the batch.

        # CFG (Classifier-Free Guidance) masking during training
        if train: # self.is_train could also be used
            # Text CFG
            text_cfg_mask = torch.rand(size=(bs,), device=device) < 0.15 # 15% unconditional for text
            encoder_text_hidden_states[text_cfg_mask] = torch.zeros_like(encoder_text_hidden_states[0]).to(device, dtype)
            # text_attention_mask might also need to be zeroed or handled for CFG items,
            # but often just zeroing hidden_states is enough if model handles zero padding.

            # Speaker CFG
            speaker_cfg_mask = torch.rand(size=(bs,), device=device) < 0.50 # 50% unconditional for speaker
            speaker_embds[speaker_cfg_mask] = torch.zeros_like(speaker_embds[0]).to(device, dtype)

            # Lyrics CFG
            lyric_cfg_mask = torch.rand(size=(bs,), device=device) < 0.15 # 15% unconditional for lyrics
            lyric_token_ids[lyric_cfg_mask] = 0 # Assuming 0 is a padding/null token
            lyric_mask[lyric_cfg_mask] = 0


        return (
            keys, # List of strings
            target_latents, # Tensor
            attention_mask, # Tensor (for latents/noisy_image)
            encoder_text_hidden_states, # Tensor
            text_attention_mask, # Tensor
            speaker_embds, # Tensor
            lyric_token_ids, # Tensor
            lyric_mask, # Tensor
            mert_ssl_hidden_states_list, # List of Tensors
            mhubert_ssl_hidden_states_list # List of Tensors
        )

    def get_scheduler(self): # Unchanged
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self): # Unchanged
        trainable_params = [p for name, p in self.transformers.named_parameters() if p.requires_grad]
        logger.info(f"Found {len(trainable_params)} trainable parameters in self.transformers.")
        if not trainable_params:
            logger.warning("No trainable parameters found in self.transformers! Check LoRA setup or model requires_grad flags.")
            # Add a dummy parameter if there are no trainable ones to prevent optimizer error,
            # though this indicates a setup problem.
            # return torch.optim.AdamW([torch.nn.Parameter(torch.randn(1))], lr=self.hparams.learning_rate)


        optimizer = torch.optim.AdamW(
            params=[{'params': trainable_params}],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 1.0 - progress)
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


    def train_dataloader(self):
        logger.info(f"Setting up train dataloader with cached dataset path: {self.hparams.dataset_path}")
        self.train_dataset = Text2MusicDataset( # Uses the modified dataset class
            train=True,
            train_dataset_path=self.hparams.dataset_path, # This now points to '/cache'
            shuffle_flag=True # DataLoader will also shuffle per epoch if shuffle=True below
        )
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check dataset_path and preprocessing.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.trainer.datamodule.batch_size if hasattr(self.trainer.datamodule, 'batch_size') else 1, # Get batch_size from trainer/datamodule if possible
            shuffle=True, # Shuffle a_getitems
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn, # Crucial: uses the new collate_fn
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32): # Unchanged
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device) # Ensure timesteps is on the correct device
        
        # Ensure timesteps are scalar or 1D for comparison
        if timesteps.ndim > 1: timesteps = timesteps.flatten()

        step_indices = []
        for t_val in timesteps:
            # Find the index where schedule_timesteps matches t_val
            # Handle cases where t_val might not be exactly in schedule_timesteps due to float precision
            # by finding the closest. Or ensure t_val is always from scheduler.timesteps.
            # The current get_timestep method ensures this.
            indices = (schedule_timesteps == t_val.item()).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                # This should not happen if timesteps are sampled correctly from self.scheduler.timesteps
                logger.error(f"Timestep value {t_val} not found in scheduler's timesteps: {schedule_timesteps}")
                # Fallback or error: find closest or raise
                # For now, let's assume it's found. If error occurs, this needs more robust handling.
                # Example fallback: use index 0, or find closest.
                # differences = torch.abs(schedule_timesteps - t_val.item())
                # step_indices.append(torch.argmin(differences).item())
                raise ValueError(f"Timestep {t_val} not in scheduler.timesteps")

            step_indices.append(indices[0].item()) # Take the first match if multiple (should be unique)
            
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device): # Unchanged
        if self.hparams.timestep_densities_type == "logit_normal":
            u = torch.normal(mean=self.hparams.logit_mean, std=self.hparams.logit_std, size=(bsz, ), device="cpu")
            u = torch.sigmoid(u) # Sigmoid to map to (0,1)
            
            # Ensure num_train_timesteps is an int and valid
            num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
            if num_train_timesteps <= 0:
                raise ValueError(f"Scheduler num_train_timesteps must be positive, got {num_train_timesteps}")

            indices = (u * num_train_timesteps).long()
            indices = torch.clamp(indices, 0, num_train_timesteps - 1)
            timesteps = self.scheduler.timesteps[indices].to(device) # Move to target device
        else:
            # Fallback or other sampling strategies
            logger.warning(f"Unsupported timestep_densities_type: {self.hparams.timestep_densities_type}. Using random uniform sampling.")
            indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device="cpu").long()
            timesteps = self.scheduler.timesteps[indices].to(device)
        return timesteps

    def run_step(self, batch, batch_idx): # Adapted for new preprocess output
        # plot_step now expects the same kind of batch if called with a dataloader batch
        if self.is_train and self.hparams.every_plot_step > 0 and self.global_step % self.hparams.every_plot_step == 0 :
             if self.local_rank == 0 : # Only plot on rank 0
                logger.info(f"Generating plot at global step {self.global_step}")
                self.plot_step(batch, batch_idx) # batch is already preprocessed for training

        (
            _keys, # keys not directly used in loss computation here
            target_latents,
            attention_mask_for_latents, # Renamed for clarity, this masks the sequence dim of latents
            encoder_text_hidden_states,
            _text_attention_mask, # text_attention_mask from batch is for the *original* text seq length
                                 # self.transformers might use it or derive its own from encoder_hidden_states
            speaker_embds,
            lyric_token_ids,
            _lyric_mask, # Similar to text_attention_mask
            mert_ssl_hidden_states_list, # List of Tensors
            mhubert_ssl_hidden_states_list,
        ) = self.preprocess(batch, train=self.is_train) # Use self.is_train for CFG control

        # Ensure tensors are on the correct device (Lightning usually handles batch)
        # This is more of a double-check if issues arise.
        current_device = self.device
        target_latents = target_latents.to(current_device)
        attention_mask_for_latents = attention_mask_for_latents.to(current_device)
        encoder_text_hidden_states = encoder_text_hidden_states.to(current_device)
        speaker_embds = speaker_embds.to(current_device)
        lyric_token_ids = lyric_token_ids.to(current_device)
        
        # For SSL lists, each tensor within the list needs to be on the correct device
        all_ssl_final_features = []
        if mert_ssl_hidden_states_list: # It's a list of tensors, one per batch item
            all_ssl_final_features.append([s.to(current_device) for s in mert_ssl_hidden_states_list if s.numel() > 0])
        if mhubert_ssl_hidden_states_list:
            all_ssl_final_features.append([s.to(current_device) for s in mhubert_ssl_hidden_states_list if s.numel() > 0])
        
        # If after filtering empty tensors, a whole SSL type list is empty, don't append it
        all_ssl_final_features = [ssl_list for ssl_list in all_ssl_final_features if ssl_list]


        target_image = target_latents # Alias for clarity, as in original code
        dtype = target_image.dtype
        
        noise = torch.randn_like(target_image, device=current_device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, current_device)

        sigmas = self.get_sd3_sigmas(timesteps=timesteps, device=current_device, n_dim=target_image.ndim, dtype=dtype)
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image
        target_for_loss = target_image # Flow matching target

        # The transformer input attention_mask should correspond to the sequence dimension of noisy_image/target_latents
        # `attention_mask_for_latents` was created based on `latent_seq_len`
        transformer_input_attention_mask = attention_mask_for_latents 


        # In Pipeline.run_step, for debugging (early in the function):
        # In Pipeline.run_step, for debugging (early in the function):
        # logger.info("--- Debugging single LoRA layer ---")
        # try:
        #     # Assuming speaker_embedder is a lora.Linear layer
        #     speaker_embedder_layer = self.transformers.speaker_embedder # type: ignore
        #     base_layer_weight = speaker_embedder_layer.base_layer.weight # type: ignore
            
        #     # Create dummy input matching the speaker_embedder's base_layer input features and dtype
        #     dummy_input_speaker = torch.randn(1, 512, device=self.device, dtype=base_layer_weight.dtype)

        #     # Check requires_grad for LoRA weights of this specific layer
        #     lora_A_weight = speaker_embedder_layer.lora_A['default'].weight # type: ignore
        #     lora_B_weight = speaker_embedder_layer.lora_B['default'].weight # type: ignore
        #     logger.info(f"speaker_embedder.lora_A['default'].weight.requires_grad: {lora_A_weight.requires_grad}")
        #     logger.info(f"speaker_embedder.lora_B['default'].weight.requires_grad: {lora_B_weight.requires_grad}")

        #     # Determine if MixedPrecisionPlugin is active and what its type is
        #     amp_plugin_active = False
        #     amp_dtype_to_use = None
        #     if hasattr(self.trainer, 'precision_plugin') and isinstance(self.trainer.precision_plugin, MixedPrecisionPlugin): # type: ignore
        #         amp_plugin_active = True
        #         if self.trainer.precision_plugin.precision == "bf16-mixed": # type: ignore
        #             amp_dtype_to_use = torch.bfloat16
        #         elif self.trainer.precision_plugin.precision == "16-mixed": # type: ignore
        #             amp_dtype_to_use = torch.float16
            
        #     logger.info(f"AMP plugin active for isolated test: {amp_plugin_active}, AMP dtype for isolated test: {amp_dtype_to_use}")

        #     # Perform forward pass under AMP context if used in training
        #     # Ensure the layer itself is in training mode for dropout, etc. (self.transformers.train() should handle this for the whole model)
        #     with torch.cuda.amp.autocast(enabled=amp_plugin_active, dtype=amp_dtype_to_use if amp_plugin_active else None):
        #         dummy_output_speaker = speaker_embedder_layer(dummy_input_speaker)
            
        #     logger.info(f"Output dtype from isolated speaker_embedder_layer test: {dummy_output_speaker.dtype}")
        #     logger.info(f"Output .grad_fn from isolated speaker_embedder_layer test: {dummy_output_speaker.grad_fn}")
        #     if dummy_output_speaker.grad_fn is None:
        #         logger.error("CRITICAL (Isolated Test): speaker_embedder output has no grad_fn!")
        #     else:
        #         logger.info("GOOD (Isolated Test): speaker_embedder output has a grad_fn.")

        # except AttributeError as ae:
        #     logger.error(f"AttributeError during single LoRA layer debug (check layer path or PEFT structure): {ae}", exc_info=True)
        # except Exception as e:
        #     logger.error(f"Error during single LoRA layer debug: {e}", exc_info=True)
        # logger.info("--- End of single LoRA layer debug ---")


        transformer_output = self.transformers(
            hidden_states=noisy_image, # x
            attention_mask=transformer_input_attention_mask, # Mask for the latent sequence dimension
            encoder_text_hidden_states=encoder_text_hidden_states, # Text condition
            text_attention_mask=batch["text_attention_mask"].to(current_device), # Use original text mask from batch for cross-attn
            speaker_embeds=speaker_embds,   # Speaker condition
            lyric_token_idx=lyric_token_ids, # Lyric condition
            lyric_mask=batch["lyric_masks"].to(current_device),       # Use original lyric mask from batch for lyric encoder
            timestep=timesteps.to(dtype=dtype), # Timestep condition (ensure dtype match)
            ssl_hidden_states=all_ssl_final_features if all_ssl_final_features else None, # SSL condition
        )
        model_pred = transformer_output.sample
        model_pred_raw = transformer_output.sample
        # logger.info(f"model_pred_raw.grad_fn: {model_pred_raw.grad_fn}")
        # if model_pred_raw.grad_fn is None:
        #     logger.warning("Raw model_pred from transformer has no grad_fn! LoRA might not be effective.")
        proj_losses = transformer_output.proj_losses # These are the SSL projection losses

        model_pred = model_pred * (-sigmas) + noisy_image # Preconditioning

        # Loss mask should also use the transformer_input_attention_mask
        # It needs to be expanded to match the shape of model_pred/target_for_loss [B, C, H, W_latent]
        # transformer_input_attention_mask is [B, W_latent]
        loss_mask_expanded = transformer_input_attention_mask.unsqueeze(1).unsqueeze(1).expand_as(target_for_loss)
        
        selected_model_pred = (model_pred * loss_mask_expanded).reshape(bsz, -1)
        selected_target = (target_for_loss * loss_mask_expanded).reshape(bsz, -1)

        denoising_loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        denoising_loss = denoising_loss.sum(dim=1) / (loss_mask_expanded.reshape(bsz, -1).sum(dim=1) + 1e-8) # Normalize by number of unmasked elements
        denoising_loss = denoising_loss.mean()


        prefix = "train" if self.is_train else "val" # Or "test" if applicable
        self.log(f"{prefix}/denoising_loss", denoising_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        total_proj_loss_value = 0.0
        if proj_losses: # proj_losses is expected to be a list of (name, loss_tensor) tuples
            num_proj_losses = 0
            for k_loss, v_loss in proj_losses:
                if isinstance(v_loss, torch.Tensor) and v_loss.numel() > 0 : # Check if it's a valid tensor loss
                    self.log(f"{prefix}/{k_loss}_loss", v_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                    total_proj_loss_value += v_loss
                    num_proj_losses +=1
            if num_proj_losses > 0:
                total_proj_loss_value = total_proj_loss_value / num_proj_losses
        
        final_loss = denoising_loss
        if isinstance(total_proj_loss_value, torch.Tensor): # Ensure it's a tensor before scaling
            final_loss = denoising_loss + total_proj_loss_value * self.hparams.ssl_coeff
        
        self.log(f"{prefix}/total_loss", final_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.is_train and self.lr_schedulers() is not None: # Log LR only during training
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log("lr", learning_rate, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        return final_loss

    def training_step(self, batch, batch_idx):
        if not batch: # Handle empty batch from collate_fn if all items were problematic
            logger.warning(f"Skipping training step for batch_idx {batch_idx} due to empty batch.")
            return None
        return self.run_step(batch, batch_idx)

    # diffusion_process for inference remains largely the same,
    # but it needs text_encoder and tokenizer if called standalone.
    # For plot_step, these will come from the preprocessed batch.
    @torch.no_grad()
    def diffusion_process( 
        self,
        duration, # Or use latent_shape if provided
        encoder_text_hidden_states, # Pre-computed
        text_attention_mask,      # Pre-computed
        speaker_embds,            # Pre-computed
        lyric_token_ids,          # Pre-computed
        lyric_mask,               # Pre-computed
        target_latent_shape=None, # Optional: to specify exact output shape
        random_generators=None,
        infer_steps=60,
        guidance_scale=7.0, # Typical default
        omega_scale=0.0, # Shift parameter for scheduler during inference, often 0 for ddim like
    ):
        # This method is for generation from scratch / conditions.
        # The inputs (text embeddings, etc.) are now expected to be pre-computed.

        do_classifier_free_guidance = guidance_scale > 1.0

        device = encoder_text_hidden_states.device if encoder_text_hidden_states is not None else self.device
        dtype = self.transformers.dtype # Use transformer's dtype

        bsz = 1
        if encoder_text_hidden_states is not None:
            bsz = encoder_text_hidden_states.shape[0]
        elif speaker_embds is not None:
            bsz = speaker_embds.shape[0]
        
        # Scheduler for inference (can be same as training or different)
        # Using a new instance for inference settings
        infer_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T, # Use T from training for consistency
            shift=self.hparams.shift, # Or a different shift for inference if desired
            # beta_schedule="linear", # Or as configured
        )
        infer_scheduler.set_timesteps(num_inference_steps=infer_steps, device=device)
        timesteps = infer_scheduler.timesteps


        if target_latent_shape is None:
            frame_length_latent = int(duration * 44100 / 512 / 8) # Default latent calculation
            # Expected latent shape [B, C, H_feat, W_feat_seq] e.g. [B, 8, 16, frame_length_latent]
            latent_channels = self.transformers.config.in_channels # Input channels to transformer
            # This needs a robust way to get latent H, W structure if not fixed.
            # Assuming fixed C=8, H_feat=16 based on previous observations
            latent_shape = (bsz, latent_channels, 16, frame_length_latent)
        else:
            latent_shape = target_latent_shape

        latents = randn_tensor(shape=latent_shape, generator=random_generators, device=device, dtype=dtype)
        
        # Create attention mask for the latents (all ones for generated content)
        # This mask is for the *sequence dimension* of the latents fed to the transformer
        latent_attention_mask_for_transformer = torch.ones(bsz, latent_shape[-1], device=device, dtype=torch.float32)


        # Prepare CFG inputs
        if do_classifier_free_guidance:
            # Text
            if encoder_text_hidden_states is not None:
                uncond_text_embeds = torch.zeros_like(encoder_text_hidden_states).to(device, dtype)
                encoder_text_hidden_states = torch.cat([uncond_text_embeds, encoder_text_hidden_states])
            if text_attention_mask is not None: # Mask for uncond is often all ones up to seq_len, or same as cond
                uncond_text_mask = torch.ones_like(text_attention_mask).to(device, text_attention_mask.dtype)
                text_attention_mask = torch.cat([uncond_text_mask, text_attention_mask])
            
            # Speaker
            if speaker_embds is not None:
                uncond_speaker_embeds = torch.zeros_like(speaker_embds).to(device, dtype)
                speaker_embds = torch.cat([uncond_speaker_embeds, speaker_embds])

            # Lyrics
            if lyric_token_ids is not None:
                uncond_lyric_ids = torch.zeros_like(lyric_token_ids).to(device, lyric_token_ids.dtype)
                lyric_token_ids = torch.cat([uncond_lyric_ids, lyric_token_ids])
            if lyric_mask is not None:
                uncond_lyric_mask = torch.zeros_like(lyric_mask).to(device, lyric_mask.dtype) # Zeros for uncond lyrics
                lyric_mask = torch.cat([uncond_lyric_mask, lyric_mask])
            
            if latent_attention_mask_for_transformer is not None:
                 latent_attention_mask_for_transformer = torch.cat([latent_attention_mask_for_transformer] * 2)


        momentum_buffer = MomentumBuffer() # If using APG guidance

        for t in tqdm(timesteps, desc="Diffusion Process"):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            current_timestep_tensor = t.expand(latent_model_input.shape[0]).to(dtype)

            # No SSL features are passed during this standard diffusion inference
            noise_pred_out = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=latent_attention_mask_for_transformer, # Mask for latent sequence dim
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask, # Mask for text sequence dim
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask, # Mask for lyric sequence dim
                timestep=current_timestep_tensor,
                ssl_hidden_states=None # No SSL conditioning during standard inference
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
                # Standard CFG guidance:
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Using APG:
                noise_pred = apg_forward(
                     pred_cond=noise_pred_text,
                     pred_uncond=noise_pred_uncond,
                     guidance_scale=guidance_scale,
                     momentum_buffer=momentum_buffer,
                )

            else:
                noise_pred = noise_pred_out
            
            # Scheduler step
            # Note: original trainer code used omega_scale for scheduler.step.
            # This is specific to FlowMatchEulerDiscreteScheduler from SD3 if it has an omega param.
            # Vanilla DDIM/DDPM schedulers don't have omega.
            # Assuming your scheduler matches the one in training.
            scheduler_output = infer_scheduler.step(model_output=noise_pred, timestep=t, sample=latents) #Removed omega for now
            latents = scheduler_output.prev_sample
        
        return latents
    
    def predict_step(self, batch, batch_idx=0): # batch_idx for completeness
        if not batch: # Handle empty batch
            logger.warning(f"Skipping predict_step for batch_idx {batch_idx} due to empty batch.")
            return None
            
        # Data from this batch is ALREADY pre-processed by the new Text2MusicDataset
        (
            keys,
            target_latents, # Ground truth latents (for comparison if any, or shape reference)
            _latent_attention_mask, # Mask for GT latents
            encoder_text_hidden_states, # Pre-computed text embeds for this batch
            text_attention_mask,      # Mask for text embeds
            speaker_embds,            # Pre-computed speaker embeds
            lyric_token_ids,          # Pre-computed lyric tokens
            lyric_mask,               # Mask for lyric tokens
            _mert_ssl_hidden_states_list, # Not used in diffusion_process for generation
            _mhubert_ssl_hidden_states_list,
        ) = self.preprocess(batch, train=False) # train=False to disable CFG dropout for conditions

        # For plot_step, we usually want to generate audio based on the prompts in the batch
        # The duration can be inferred from target_latents' shape or a fixed value.
        # Example: Use target_latents' shape to determine generation shape.
        # target_latents shape: [B, C, H, W_latent_seq]
        
        bsz = target_latents.shape[0]
        target_gen_shape = target_latents.shape 
        
        # If you want a fixed duration for plotting, e.g., 10 seconds:
        # plot_duration = 10 # seconds
        # Or use original duration if available and not too long
        # For now, let's use the shape of the ground truth latents from batch for consistency
        # This means duration is implicitly handled by target_gen_shape[-1]

        random_generators = [torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32 - 1) + self.global_step + idx) for idx in range(bsz)]
        seeds = [gen.initial_seed() for gen in random_generators]

        # Use the pre-computed embeddings from the batch for diffusion_process
        pred_latents = self.diffusion_process(
            duration=None, # Not needed if target_latent_shape is given
            target_latent_shape=target_gen_shape,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=random_generators,
            infer_steps=60, # Or a hyperparameter
            guidance_scale=7.0, # Typical inference guidance scale
            omega_scale=0.0
        )

        # Decode predicted latents to waveform
        # The 'wav_lengths' in batch are for the original audio, not directly for decode if latents changed length.
        # dcae.decode might not need audio_lengths if it decodes the full latent.
        # If dcae.decode needs a target length, derive it from pred_latents shape or original audio.
        # Let's assume dcae.decode can handle the latents directly.
        _sr, pred_wavs_list = self.dcae.decode(pred_latents, sr=48000) # Returns list of tensors
        
        # The batch also contains ground truth for comparison (target_wavs was NOT in cached data)
        # We need to decode target_latents from the batch to get comparable target_wavs
        _sr_target, target_wavs_list_decoded = self.dcae.decode(target_latents, sr=48000)


        return {
            # "target_wavs": batch.get("target_wavs_from_raw_audio"), # If we were to pass raw audio for comparison
            "target_wavs_decoded": target_wavs_list_decoded, # Decoded from cached target_latents
            "pred_wavs": pred_wavs_list, # Decoded from newly generated pred_latents
            "keys": keys,
            "prompts": batch["prompts"], # list of strings from dataset
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"], # from dataset for plot_step
            "sr": _sr, # sample rate from dcae.decode
            "seeds": seeds,
        }

    def construct_lyrics(self, candidate_lyric_chunk_for_item): # Unchanged
        # candidate_lyric_chunk_for_item is now a list of dicts for a single item.
        lyrics = []
        if candidate_lyric_chunk_for_item: # Check if it's not None or empty
            for chunk in candidate_lyric_chunk_for_item:
                lyrics.append(chunk["lyric"])
        return "\n".join(lyrics) if lyrics else "[No lyrics provided]"


    def plot_step(self, batch, batch_idx): # Unchanged in principle, but uses predict_step's output
        # This condition should ideally be handled by PyTorch Lightning's every_n_train_steps for callbacks
        # or by `self.log_every_n_steps` for general logging.
        # For explicit plotting, this check is fine.
        # global_step = self.global_step
        # if not self.is_train or self.hparams.every_plot_step <= 0 or global_step % self.hparams.every_plot_step != 0:
        #     return
        # if self.local_rank != 0: # Only plot on rank 0
        #     return
        # The above logic is good, just ensuring it's not redundant with PL's own step controls.
        # The original code's check is fine.

        logger.info(f"Plotting at global step: {self.global_step}, local rank: {self.local_rank}")
        results = self.predict_step(batch, batch_idx)
        if results is None: # predict_step might return None if batch was empty
            logger.warning("predict_step returned None, skipping plot.")
            return

        # target_wavs are now decoded from cached target_latents
        target_wavs_list = results["target_wavs_decoded"]
        pred_wavs_list = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"] # This is now a list of strings
        candidate_lyric_chunks_batch = results["candidate_lyric_chunks"] # This is a list of lists of dicts
        sr = results["sr"]
        seeds = results["seeds"]
        
        # Ensure logger is available (usually is in LightningModule)
        if self.logger is None or self.logger.experiment is None:
            logger.warning("Logger not available, cannot save plot results.")
            return

        log_dir = self.logger.log_dir if hasattr(self.logger, 'log_dir') else 'lightning_logs/unknown_version'
        save_dir_base = Path(log_dir) / "eval_results"
        save_dir_step = save_dir_base / f"step_{self.global_step}"
        
        try:
            if not save_dir_step.exists():
                os.makedirs(save_dir_step, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create save directory {save_dir_step}: {e}")
            return

        # Iterate through the batch
        for i in range(len(keys)):
            key_val = keys[i]
            prompt_val = prompts[i] # Now a string
            # candidate_lyric_chunk_val is a list of dicts for the i-th item
            candidate_lyric_chunk_val = candidate_lyric_chunks_batch[i] if candidate_lyric_chunks_batch and i < len(candidate_lyric_chunks_batch) else []
            seed_val = seeds[i]
            
            target_wav_tensor = target_wavs_list[i] # Tensor [C, T]
            pred_wav_tensor = pred_wavs_list[i]   # Tensor [C, T]

            lyric_str = self.construct_lyrics(candidate_lyric_chunk_val)
            key_prompt_lyric = f"# KEY\n\n{key_val}\n\n\n# PROMPT\n\n{prompt_val}\n\n\n# LYRIC\n\n{lyric_str}\n\n# SEED\n\n{seed_val}\n\n"
            
            try:
                # Sanitize key_val for filename
                safe_key = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in str(key_val)).rstrip()
                safe_key = safe_key.replace(' ', '_')
                
                target_filename = save_dir_step / f"target_wav_{safe_key}_{i}.flac"
                pred_filename = save_dir_step / f"pred_wav_{safe_key}_{i}.flac"
                text_filename = save_dir_step / f"info_{safe_key}_{i}.txt"

                torchaudio.save(str(target_filename), target_wav_tensor.float().cpu(), sr)
                torchaudio.save(str(pred_filename), pred_wav_tensor.float().cpu(), sr)
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(key_prompt_lyric)
            except Exception as e:
                logger.error(f"Error saving plot files for key {key_val}, item {i}: {e}", exc_info=True)

def merge_lora_checkpoint(lora_ckpt_path: str, output_dir: str) -> str | None:
    """
    Merges LoRA weights from a checkpoint into the base model and saves a new checkpoint with updated state_dict.

    Args:
        lora_ckpt_path (str): Path to the checkpoint with LoRA weights.
        output_dir (str): Directory to save the merged checkpoint.

    Returns:
        str | None: Path to the merged checkpoint or None if merging fails.
    """
    logger.info(f"Merging LoRA weights from checkpoint: {lora_ckpt_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the full original checkpoint dictionary
    try:
        original_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        logger.info("Loaded full original checkpoint dictionary.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None

    # Step 2: Extract hyperparameters and create a new Pipeline instance
    hparams = original_checkpoint.get('hyper_parameters', {})
    try:
        # Pass required arguments explicitly, using defaults if not in hparams
        model = Pipeline(
            learning_rate=hparams.get('learning_rate', 1e-4),
            num_workers=hparams.get('num_workers', 4),
            train=hparams.get('train', True),
            T=hparams.get('T', 1000),
            weight_decay=hparams.get('weight_decay', 1e-2),
            every_plot_step=hparams.get('every_plot_step', 2000),
            shift=hparams.get('shift', 3.0),
            logit_mean=hparams.get('logit_mean', 0.0),
            logit_std=hparams.get('logit_std', 1.0),
            timestep_densities_type=hparams.get('timestep_densities_type', 'logit_normal'),
            ssl_coeff=hparams.get('ssl_coeff', 1.0),
            checkpoint_dir=hparams.get('checkpoint_dir', None),
            max_steps=hparams.get('max_steps', 200000),
            warmup_steps=hparams.get('warmup_steps', 4000),
            dataset_path=hparams.get('dataset_path', './cache'),
            lora_config_path=hparams.get('lora_config_path', None),

            new_lora_config_path    = args.new_lora_config_path,
            new_lora_name           = args.new_lora_name,
            freeze_prev_loras       = args.freeze_prev_loras,
        )
        logger.info("Created new Pipeline instance with checkpoint hyperparameters.")
    except Exception as e:
        logger.error(f"Failed to create Pipeline instance: {e}")
        return None

    # Step 3: Load the original state_dict into the model
    try:
        model.load_state_dict(original_checkpoint['state_dict'])
        logger.info("Loaded original state_dict into the model.")
    except Exception as e:
        logger.error(f"Failed to load state_dict: {e}")
        return None

    # Step 4: Merge LoRA weights if applicable
    if hasattr(model.transformers, 'fuse_lora'):
        try:
            model.transformers.fuse_lora()
            logger.info("Fused LoRA weights into the base model using 'fuse_lora'.")
        except Exception as e:
            logger.error(f"Error fusing LoRA weights: {e}")
            return None
    elif hasattr(model.transformers, 'merge_and_unload'):
        try:
            merged_model = model.transformers.merge_and_unload()
            model.transformers = merged_model
            logger.info("Merged LoRA weights using 'merge_and_unload'.")
        except Exception as e:
            logger.error(f"Error merging LoRA weights: {e}")
            return None
    else:
        logger.error("Model does not have 'fuse_lora' or 'merge_and_unload' method; cannot merge LoRA weights.")
        return None

    # Step 5: Update the original checkpoint's state_dict with the merged model's state_dict
    original_checkpoint['state_dict'] = model.state_dict()

    # Step 6: Save the updated checkpoint
    output_ckpt_path = output_path / "merged_model.ckpt"
    try:
        torch.save(original_checkpoint, output_ckpt_path)
        logger.info(f"Saved merged checkpoint to: {output_ckpt_path}")
    except Exception as e:
        logger.error(f"Failed to save merged checkpoint: {e}")
        return None

    return str(output_ckpt_path)


def main(args):
    # Ensure dataset_path defaults to the cache if not specified,
    # or user provides the path to the cache.
    if args.dataset_path == "./data/your_dataset_path": # Original default
        args.dataset_path = "./cache" # New default for cached data
        logger.info(f"Dataset path not specified or default, using cached data path: {args.dataset_path}")

    # Batch size needs to be obtained for DataLoader.
    # PyTorch Lightning Trainer usually manages this. If running standalone or if DataModule isn't used,
    # we might need to add a batch_size arg to script or Pipeline.
    # For now, assuming Trainer will inject it or DataLoader will use its own default.
    # Let's add a batch_size argument for clarity.

    # Handle LoRA merging if --merge-lora is specified
    ckpt_path_for_training = args.ckpt_path
    if args.merge_lora and args.ckpt_path:
        temp_dir = Path("./temp_merged_ckpt")
        temp_dir.mkdir(parents=True, exist_ok=True)
        merged_ckpt_path = merge_lora_checkpoint(args.ckpt_path, str(temp_dir))
        if merged_ckpt_path is None:
            logger.error("Failed to merge LoRA checkpoint. Exiting.")
            exit(1)
        ckpt_path_for_training = merged_ckpt_path
        logger.info(f"Using merged checkpoint for training: {ckpt_path_for_training}")
    torch.cuda.empty_cache()  # Add this line

    model = Pipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        ssl_coeff=args.ssl_coeff,
        T=args.T,
        weight_decay=args.weight_decay,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        timestep_densities_type=args.timestep_densities_type,
        warmup_steps=args.warmup_steps,
        lora_config_path=args.lora_config_path,

        new_lora_config_path    = args.new_lora_config_path,
        new_lora_name           = args.new_lora_name,
        freeze_prev_loras       = args.freeze_prev_loras,
    )
    
    # The batch size for the DataLoader will be implicitly set by the Trainer
    # via `trainer.fit(model)` if no DataModule is used.
    # If `model.train_dataloader()` is called directly, it would need a batch_size.
    # In PL, `Trainer(devices=..., strategy=...)` also influences effective batch size.
    # We can add batch_size to Pipeline hparams if needed, or assume PL handles it.
    # The current `train_dataloader` tries to get it from `self.trainer.datamodule.batch_size`
    # or defaults to 1. This needs a batch_size arg if not using a datamodule.


    checkpoint_callback = ModelCheckpoint(
        monitor=None, # Not monitoring a specific val metric for saving best
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1, # Save all checkpoints matching every_n_train_steps
        save_last=False # Optionally save the last checkpoint
    )
    
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version_name = f"{current_time_str}_{args.exp_name}" if args.exp_name else current_time_str
    
    logger_callback = TensorBoardLogger(
        save_dir=args.logger_dir,
        name="", # Log directly into version_name folder
        version=version_name
    )
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        num_nodes=args.num_nodes,
        precision=args.precision if torch.cuda.is_available() else "32-true", # Ensure precision matches device capability
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=args.strategy if torch.cuda.is_available() else "auto", # deepspeed needs cuda
        max_epochs=args.epochs if args.epochs > 0 else -1, # Default to steps if epochs is -1
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps, # Added arg
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        # reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs, # Deprecated, use every_n_epochs in DataLoader
        val_check_interval=args.val_check_interval if args.val_check_interval is not None else 1.0, # Default to check every epoch if not set
        # Add batch_size to trainer if not using datamodule
        # This is usually handled by trainer.fit(model, train_dataloaders=...) or datamodule
    )

    # The batch size for the DataLoader created in model.train_dataloader() needs to be set.
    # Pytorch Lightning usually gets this from a DataModule or a `batch_size` argument in `Trainer.fit`.
    # If you don't use a DataModule, you might need to pass `batch_size` to your `Pipeline`
    # or set `model.trainer.datamodule.batch_size = args.batch_size` before `trainer.fit` if that's how it's accessed.
    # A cleaner way is to add `batch_size` to your `Pipeline` hparams and use it in `train_dataloader`.

    logger.info(f"Starting training with dataset from: {args.dataset_path}")
    trainer.fit(
        model,
        # ckpt_path=args.ckpt_path,
        ckpt_path=ckpt_path_for_training
        # train_dataloaders=model.train_dataloader() # Optionally pass dataloader if batch size needs explicit control here
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Existing args from original script
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4) # Reduced default as per typical local setup
    parser.add_argument("--epochs", type=int, default=-1, help="Number of epochs, -1 for max_steps controlled.")
    parser.add_argument("--max_steps", type=int, default=10000000) # Reduced for quicker local test
    parser.add_argument("--every_n_train_steps", type=int, default=500) # Checkpoint saving frequency
    parser.add_argument("--dataset_path", type=str, default="./cache", help="Path to the PREPROCESSED dataset.") # MODIFIED DEFAULT
    parser.add_argument("--exp_name", type=str, default="text2music_cached_test")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision, e.g., '32-true', '16-mixed', 'bf16-mixed'")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs per node, or list of GPU IDs.")
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for ACEStepPipeline and HF model cache.")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0) # Adjusted default
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    # parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1) # Deprecated
    parser.add_argument("--every_plot_step", type=int, default=500) # Plotting frequency
    parser.add_argument("--val_check_interval", type=float, default=0.0, help="Validation check interval (float for fraction of epoch, int for steps).")
    parser.add_argument("--lora_config_path", type=str, default=None)

    # Add hparams from Pipeline.__init__ not already in parser
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--timestep_densities_type", type=str, default="logit_normal")
    parser.add_argument("--ssl_coeff", type=float, default=0.5) # Adjusted default based on common practice
    parser.add_argument("--warmup_steps", type=int, default=1000) # Adjusted for potentially smaller total steps

    # Add new args for Trainer/DataLoader
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device.") # ADDED
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (e.g., 'ddp', 'deepspeed_stage_2'). 'auto' is good default.")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Logging frequency.")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights from checkpoint before training.")

    # trainer.py  (parser section – add directly after --lora_config_path)
    parser.add_argument("--new_lora_config_path", type=str, default=None,
                        help="JSON file that describes the *second* LoRA you want to train")
    parser.add_argument("--new_lora_name",      type=str, default="lora2",
                        help="Name to register the new adapter inside PEFT")
    parser.add_argument("--freeze_prev_loras",  action="store_true",
                        help="Freeze all existing LoRA adapters that come with --ckpt_path")


    args = parser.parse_args()
    
    # A way to pass batch_size to the model if train_dataloader needs it and no datamodule used
    # This is a bit of a workaround. A cleaner way is to include batch_size in Pipeline's hparams
    # and use self.hparams.batch_size in train_dataloader.
    # For now, we'll assume the user might set an environment variable or rely on PL defaults for batch size if not using datamodule.
    # Or, modify train_dataloader to accept batch_size from self.hparams if added there.
    # Let's add it to Pipeline's hparams.
    
    # Add batch_size to args that Pipeline will receive if it's not already an hparam.
    # It's better to add batch_size to Pipeline's __init__ signature.
    # For a quick fix, we can make train_dataloader use args.batch_size.
    # This requires `args` to be accessible in `train_dataloader` or `batch_size` to be an hparam.

    # Let's assume `batch_size` will be an hparam of Pipeline for cleaner integration.
    # We will modify Pipeline.__init__ and train_dataloader for this.

    main(args)