#!/usr/bin/env python3
"""
Extract LoRA weights for **one specific adapter** from a Lightning/HF checkpoint
and save them in a ComfyUI‑compatible *.safetensors* file.

The script now accepts **--adapter-name** (default: "default").
Only weights whose keys end with

    .lora_{A|B}.<adapter-name>.weight

are exported.  This lets you keep several LoRAs inside a single checkpoint and
pick exactly which one you want to convert.

We still skip lyric_encoder and speaker_embedder adapters and focus on the
UNet/transformer blocks.
"""
import argparse
import logging
import sys
import torch
from safetensors.torch import save_file

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def extract_ckpt_state_dict(ckpt_path: str, map_location: str = "cpu") -> dict:
    """Load *just* the state_dict part of a Lightning / HF checkpoint."""
    logging.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            logging.debug("Found 'state_dict' key in checkpoint root.")
            return ckpt["state_dict"]
        if "model" in ckpt:  # some PEFT saves
            logging.debug("Found 'model' key in checkpoint root.")
            return ckpt["model"]
        # assume the dict *is* the state_dict
        return ckpt

    if isinstance(ckpt, torch.nn.Module):
        logging.warning("Checkpoint appears to be an nn.Module ‑ taking its state_dict().")
        return ckpt.state_dict()

    raise RuntimeError(f"Unrecognised checkpoint format: root object is {type(ckpt)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LoRA weights for a chosen adapter and save as ComfyUI .safetensors"
    )
    parser.add_argument("--ckpt", required=True, help="Path to the .ckpt file with LoRA(s)")
    parser.add_argument("--output", required=True, help="Destination *.safetensors* file")
    parser.add_argument(
        "--adapter-name",
        default="default",
        help="Name of the adapter inside PEFT (default: 'default')",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Precision to store weights (default: fp16)",
    )
    parser.add_argument("--device", default="cpu", help="Device used for loading the checkpoint")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # ── logging setup ──────────────────────────────────────────────────────────
    lvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=lvl, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info("Adapter to extract: %s", args.adapter_name)

    # dtype to save
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map[args.precision]

    # load checkpoint
    state_dict = extract_ckpt_state_dict(args.ckpt, map_location=args.device)

    suffix_A = f".lora_A.{args.adapter_name}.weight"
    suffix_B = f".lora_B.{args.adapter_name}.weight"

    extracted: dict[str, torch.Tensor] = {}
    stats = {
        "total_pattern": 0,
        "skipped_lyric": 0,
        "skipped_speaker": 0,
        "skipped_other": 0,
    }

    for key, tensor in state_dict.items():
        lora_type_suffix: str | None = None
        if key.endswith(suffix_A):
            lora_type_suffix = ".lora_down.weight"
            original_path = key[: -len(suffix_A)]
        elif key.endswith(suffix_B):
            lora_type_suffix = ".lora_up.weight"
            original_path = key[: -len(suffix_B)]
        else:
            continue

        stats["total_pattern"] += 1

        # decide whether to keep
        if original_path.startswith("transformers.transformer_blocks"):
            comfy_prefix = "lora_unet"
            to_sanitize = original_path[len("transformers.") :]
        elif original_path.startswith("transformers.lyric_encoder"):
            stats["skipped_lyric"] += 1
            continue
        elif original_path.startswith("transformers.speaker_embedder"):
            stats["skipped_speaker"] += 1
            continue
        else:
            stats["skipped_other"] += 1
            continue

        sanitized = to_sanitize.replace(".", "_")
        comfy_key = f"{comfy_prefix}_{sanitized}{lora_type_suffix}"
        logging.debug("map %-120s -> %s", key, comfy_key)
        extracted[comfy_key] = tensor.to(save_dtype).cpu()

    # ── summary ───────────────────────────────────────────────────────────────
    logging.info("Processed %d LoRA tensors for adapter '%s'", stats["total_pattern"], args.adapter_name)
    logging.info("  kept:   %d", len(extracted))
    logging.info("  lyric:  %d skipped", stats["skipped_lyric"])
    logging.info("  speaker:%d skipped", stats["skipped_speaker"])
    logging.info("  other:  %d skipped", stats["skipped_other"])

    if not extracted:
        logging.error("Nothing extracted – check adapter name or model structure.")
        sys.exit(1)

    logging.info("Saving %d tensors to %s (dtype=%s)", len(extracted), args.output, args.precision)
    save_file(extracted, args.output, metadata={"format": "pt"})
    logging.info("✅ Done.")


if __name__ == "__main__":
    main()
