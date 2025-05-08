import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7865)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--share", type=bool, default=False)
parser.add_argument("--bf16", type=bool, default=True)
parser.add_argument("--torch_compile", type=bool, default=False)
parser.add_argument("--lora_config_path", type=str, default="data/mmx/lora_config.json", help="Path to LoRA config JSON file, if LoRA was used for training.")
parser.add_argument("--ckpt", type=str, default=None, help="Path to the PyTorch Lightning checkpoint file (e.g., last.ckpt) with trained weights.")

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)


from ui.components import create_main_demo_ui
from pipeline_ace_step import ACEStepPipeline
from data_sampler import DataSampler

from loguru import logger # Used by pipeline_ace_step.py and good for consistency
import torch


def main(args):

    model_demo = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile
    )

    # The __call__ method of ACEStepPipeline will ensure load_checkpoint is called if self.loaded is False.
    # If you want to be explicit or ensure it's loaded before the PL checkpoint logic:
    if not model_demo.loaded:
        logger.info(f"Explicitly loading ACEStepPipeline base model from: {model_demo.checkpoint_dir}")
        model_demo.load_checkpoint(model_demo.checkpoint_dir)

    # Load fine-tuned weights from PyTorch Lightning checkpoint
    # Assuming args.ckpt is the argument for your PL checkpoint path
    if args.ckpt: # Or args.pl_checkpoint_path if you used that name
        if os.path.exists(args.ckpt):
            logger.info(f"Loading fine-tuned weights from PyTorch Lightning checkpoint: {args.ckpt}")
            try:
                device_to_load = model_demo.device

                # === APPLY LoRA CONFIGURATION (IF PROVIDED) ===
                if args.lora_config_path:
                    if os.path.exists(args.lora_config_path):
                        logger.info(f"Applying LoRA config from: {args.lora_config_path} to ace_step_transformer in app.py")
                        try:
                            from peft import LoraConfig, PeftModel # PeftModel might not be needed if just add_adapter
                            import json # Make sure json is imported
                        except ImportError:
                            logger.error("PEFT library or json not found. Please install peft: pip install peft")
                            raise # Re-raise to stop execution if PEFT is critical

                        with open(args.lora_config_path, encoding="utf-8") as f:
                            lora_config_dict = json.load(f)
                        lora_config = LoraConfig(**lora_config_dict)

                        # Ensure ace_step_transformer is on the correct device before adding adapter
                        model_demo.ace_step_transformer.to(device_to_load)

                        # If model was already a PeftModel and you want to load a new adapter or set one active:
                        # This path is less likely if ACEStepTransformer2DModel.from_pretrained loads a base model.
                        # if isinstance(model_demo.ace_step_transformer, PeftModel):
                        #    model_demo.ace_step_transformer.add_adapter("default_trained", lora_config)
                        #    model_demo.ace_step_transformer.set_adapter("default_trained")
                        # else:
                        #    model_demo.ace_step_transformer = PeftModel(model_demo.ace_step_transformer, lora_config, "default_trained")

                        # More common: if model_demo.ace_step_transformer is a base Hugging Face model
                        # and trainer.py used .add_adapter()
                        if hasattr(model_demo.ace_step_transformer, "add_adapter"):
                            model_demo.ace_step_transformer.add_adapter(lora_config, adapter_name="default") # Use "default" or your specific adapter name
                            logger.info("LoRA adapter 'default' applied to ace_step_transformer in app.py using add_adapter.")
                            # Some PEFT versions/models might need explicit activation
                            if hasattr(model_demo.ace_step_transformer, 'set_adapter'):
                                model_demo.ace_step_transformer.set_adapter("default")
                        else:
                            # Fallback if add_adapter is not directly on the model (less common for this workflow)
                            from peft import get_peft_model
                            model_demo.ace_step_transformer = get_peft_model(model_demo.ace_step_transformer, lora_config)
                            logger.info("LoRA adapter applied using get_peft_model.")


                        logger.info(f"LoRA adapter applied. Current model class: {type(model_demo.ace_step_transformer)}")

                    else:
                        logger.warning(f"LoRA config path specified but not found: {args.lora_config_path}. Proceeding without applying LoRA to app model. Weight loading might largely fail.")
                # === END OF APPLY LoRA CONFIGURATION ===

                pl_ckpt_data = torch.load(args.ckpt, map_location=device_to_load)

                if 'state_dict' not in pl_ckpt_data:
                    logger.error(f"'state_dict' not found in PyTorch Lightning checkpoint: {args.ckpt}")
                else:
                    transformer_state_dict = {}
                    expected_prefix = "transformers."
                    found_weights = False
                    for k, v in pl_ckpt_data['state_dict'].items():
                        if k.startswith(expected_prefix):
                            new_key = k.replace(expected_prefix, "", 1)
                            transformer_state_dict[new_key] = v
                            found_weights = True

                    if found_weights:
                        # Now that model_demo.ace_step_transformer is LoRA-adapted (if lora_config_path was provided),
                        # the keys from the checkpoint should match the (now PEFT-modified) model structure.
                        missing_keys, unexpected_keys = model_demo.ace_step_transformer.load_state_dict(transformer_state_dict, strict=False)
                        if missing_keys:
                            logger.warning(f"After LoRA (if any), Missing keys when loading PL checkpoint: {missing_keys}")
                        if unexpected_keys:
                            logger.warning(f"After LoRA (if any), Unexpected keys when loading PL checkpoint: {unexpected_keys}")
                        logger.info("Successfully called load_state_dict for fine-tuned transformer weights.")
                        model_demo.ace_step_transformer.eval()
                    else:
                        logger.warning(f"No weights with prefix '{expected_prefix}' found in {args.ckpt}. Using base weights for ace_step_transformer.")
            except Exception as e:
                logger.error(f"Error loading weights from PyTorch Lightning checkpoint {args.ckpt}: {e}", exc_info=True)
        else:
            logger.warning(f"PyTorch Lightning checkpoint file not found: {args.ckpt}. Using base weights for ace_step_transformer.")
    else:
        logger.info("No PyTorch Lightning checkpoint (--ckpt or --pl_checkpoint_path) provided. Using base weights for ace_step_transformer.")


    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main(args)
