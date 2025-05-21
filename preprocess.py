# preprocess.py (Ensure this is the version you run)

import argparse
import os
from pathlib import Path
import torch
import torchaudio
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import numpy as np
import re
from loguru import logger
import torch.nn.functional as F

from datasets import load_from_disk, Dataset, Features, Value, Sequence, Array2D, Array3D


# Import base models and components needed
from pipeline_ace_step import ACEStepPipeline 
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from language_segmentation import LangSegment

lang_segment_global = None
lyric_tokenizer_global = None

SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, "pt": 286, "pl": 294, 
    "tr": 295, "ru": 267, "cs": 293, "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, 
    "hu": 5753, "ko": 6152, "hi": 6680
}
structure_pattern = re.compile(r"\[.*?\]")
from datasets import Dataset, concatenate_datasets

def build_dataset_in_shards(examples, shard_size=25):
    shards = []
    for start in range(0, len(examples), shard_size):
        shard = Dataset.from_list(examples[start:start + shard_size])
        shards.append(shard)
    return concatenate_datasets(shards, axis=0)
def get_lang_preprocess(text):
    global lang_segment_global
    if lang_segment_global is None:
        lang_segment_global = LangSegment()
        lang_segment_global.setfilters([
            'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 
            'da', 'de', 'dz', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 
            'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 
            'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 
            'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 
            'pt', 'qu', 'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 
            'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
        ])
    language = "en"
    langs_segments = []
    lang_counts_list = []
    try:
        if text and text.strip():
            langs_segments = lang_segment_global.getTexts(text)
            lang_counts_list = lang_segment_global.getCounts()
            if lang_counts_list and lang_counts_list[0][0]:
                language = lang_counts_list[0][0]
                if len(lang_counts_list) > 1 and language == "en" and lang_counts_list[1][0]:
                    language = lang_counts_list[1][0]
            elif not lang_counts_list :
                language = "en"
        else:
            language = "en"
    except Exception as e:
        logger.warning(f"Language segmentation for '{text[:50] if text else '[EMPTY]'}' failed: {e}")
        language = "en"
    return language, langs_segments, lang_counts_list

def tokenize_lyrics_preprocess(lyrics_text):
    global lyric_tokenizer_global
    if lyric_tokenizer_global is None:
        lyric_tokenizer_global = VoiceBpeTokenizer()
    if not lyrics_text or not lyrics_text.strip(): return [0]
    lyric_token_idx_list = [261]
    _, langs_segments, lang_counter = get_lang_preprocess(lyrics_text)
    most_common_lang = "en"
    if lang_counter and lang_counter[0][0]: most_common_lang = lang_counter[0][0]
    if most_common_lang not in SUPPORT_LANGUAGES: most_common_lang = "en"

    for lang_seg_info in langs_segments:
        lang, text_segment = lang_seg_info["lang"], lang_seg_info["text"]
        if lang not in SUPPORT_LANGUAGES: lang = "en"
        if "zh" in lang: lang = "zh"
        for line in text_segment.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                if not lyric_token_idx_list or lyric_token_idx_list[-1] != 2: lyric_token_idx_list.append(2)
                continue
            try:
                token_idx_for_line = []
                if structure_pattern.match(line_stripped):
                    token_idx_for_line = lyric_tokenizer_global.encode(line_stripped, "en")
                else:
                    token_idx_for_line = lyric_tokenizer_global.encode(line_stripped, most_common_lang)
                    if 1 in token_idx_for_line and lang != most_common_lang and lang in SUPPORT_LANGUAGES:
                         token_idx_segment_lang = lyric_tokenizer_global.encode(line_stripped, lang)
                         if token_idx_segment_lang.count(1) < token_idx_for_line.count(1) or \
                            all(t == 1 for t in token_idx_for_line):
                             token_idx_for_line = token_idx_segment_lang
                lyric_token_idx_list.extend(token_idx_for_line)
                if lyric_token_idx_list and lyric_token_idx_list[-1] != 2: lyric_token_idx_list.append(2) 
            except Exception as e:
                logger.warning(f"Lyric tokenization error for '{line_stripped}': {e}. Using UNK.")
                lyric_token_idx_list.append(1)
                if lyric_token_idx_list and lyric_token_idx_list[-1] != 2: lyric_token_idx_list.append(2)
    if len(lyric_token_idx_list) > 1 and lyric_token_idx_list[-1] == 2: lyric_token_idx_list.pop()
    return lyric_token_idx_list if lyric_token_idx_list else [0]

def static_get_text_embeddings(texts_list, tokenizer, text_encoder_model, device, text_max_length=256):
    text_encoder_model.to(device) 
    inputs = tokenizer(texts_list, return_tensors="pt", padding="max_length", truncation=True, max_length=text_max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = text_encoder_model(**inputs)
    return outputs.last_hidden_state, inputs["attention_mask"]

def static_infer_ssl_features(
    target_wavs_48k_batch, wav_lengths_48k_batch, ssl_model, resampler, device,
    target_sr, chunk_duration_sec, feature_stride_factor=320 
    ):
    ssl_model.to(device); resampler.to(device)
    resampled_wavs_batch = resampler(target_wavs_48k_batch.mean(dim=1))
    bsz = target_wavs_48k_batch.shape[0]
    sr_factor = 48000 // target_sr
    actual_lengths_target_sr = wav_lengths_48k_batch // sr_factor
    normalized_wavs_list = []
    for i in range(bsz):
        segment = resampled_wavs_batch[i, :actual_lengths_target_sr[i]]
        if segment.numel() == 0:
            norm_full = torch.zeros_like(resampled_wavs_batch[i])
        else:
            mean, var = segment.mean(), segment.var()
            norm_seg = (segment - mean) / torch.sqrt(var + 1e-7)
            norm_full = F.pad(norm_seg, (0, resampled_wavs_batch.shape[1] - actual_lengths_target_sr[i]))
        normalized_wavs_list.append(norm_full)
    
    resampled_norm_batch = torch.stack(normalized_wavs_list) if normalized_wavs_list else torch.empty(0, resampled_wavs_batch.shape[1], device=device)
    chunk_samples = target_sr * chunk_duration_sec
    batch_outputs = []

    for i in range(bsz):
        audio_full, actual_len = resampled_norm_batch[i], actual_lengths_target_sr[i]
        chunks, chunk_actual_lengths = [], []
        for start in range(0, actual_len.item(), chunk_samples):
            end = min(start + chunk_samples, actual_len.item())
            chunk_data = audio_full[start:end]
            chunk_len = len(chunk_data)
            padded_chunk = F.pad(chunk_data, (0, chunk_samples - chunk_len)) if chunk_len < chunk_samples else chunk_data
            chunks.append(padded_chunk); chunk_actual_lengths.append(chunk_len)
        if not chunks: chunks.append(torch.zeros(chunk_samples, device=device)); chunk_actual_lengths.append(0)
            
        chunk_tensor = torch.stack(chunks)
        chunk_features = ssl_model(chunk_tensor).last_hidden_state
        trimmed_item_features = []
        for j in range(chunk_features.shape[0]):
            num_feats = (chunk_actual_lengths[j] + feature_stride_factor -1) // feature_stride_factor
            if num_feats > 0: trimmed_item_features.append(chunk_features[j, :num_feats, :])
        
        if trimmed_item_features: batch_outputs.append(torch.cat(trimmed_item_features, dim=0))
        else: batch_outputs.append(torch.empty(0, ssl_model.config.hidden_size, device=device))
    return batch_outputs

def preprocess_item_data_static(item_data, models_dict, device, audio_root_abs_path, max_duration_seconds):
    key = item_data["keys"]; audio_filename = item_data["filename"]
    audio_file_path = audio_root_abs_path / audio_filename
    try:
        wav_tensor, sr = torchaudio.load(str(audio_file_path))
    except Exception as e: logger.error(f"SKIPPING load: {key} ({audio_filename}): {e}"); return None
    
    original_wav_length = wav_tensor.shape[-1]
    if wav_tensor.shape[0] == 1: wav_tensor = torch.cat([wav_tensor, wav_tensor], dim=0)
    wav_tensor = wav_tensor[:2]
    
    if sr != 48000:
        try:
            resample_48k = torchaudio.transforms.Resample(sr, 48000).to(device)
            wav_48k = resample_48k(wav_tensor.to(device)).cpu()
        except Exception as e: logger.error(f"SKIPPING resample: {key} ({audio_filename}): {e}"); return None
    else: wav_48k = wav_tensor
            
    wav_48k = torch.clamp(wav_48k, -1.0, 1.0)
    min_len_48k = 48000 * 3
    if wav_48k.shape[-1] < min_len_48k: wav_48k = F.pad(wav_48k, (0, min_len_48k - wav_48k.shape[-1]), 'constant', 0)
    
    max_len_48k = int(max_duration_seconds * 48000)
    if wav_48k.shape[-1] > max_len_48k: wav_48k = wav_48k[:, :max_len_48k]
    current_wav_len_48k = wav_48k.shape[-1]

    wav_48k_batch = wav_48k.unsqueeze(0).to(device)
    wav_lens_48k_batch = torch.tensor([current_wav_len_48k], device=device, dtype=torch.long)
    processed_data = {}

    try:
        amp_dtype = torch.float32
        if device.type == 'cuda': amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=device.type=='cuda', dtype=amp_dtype):
            dcae = models_dict['dcae'].to(device)
            latents_b, _ = dcae.encode(wav_48k_batch, wav_lens_48k_batch)
            latents = latents_b.squeeze(0)
            processed_data["target_latents"] = latents.cpu().float().numpy().astype(np.float16)
            processed_data["latent_seq_len"] = latents.shape[-1]

            tags = item_data.get("tags", [])
            #add 
            recap = item_data.get("recaption", {}).get("default", "")
            prompt = recap if recap else (", ".join(tags) if tags else "music")
            processed_data["processed_prompt_text"] = prompt[:256]

            text_embeds_b, text_mask_b = static_get_text_embeddings(
                [processed_data["processed_prompt_text"]], models_dict['text_tokenizer'], models_dict['text_encoder_model'], device)
            processed_data["encoder_text_hidden_states"] = text_embeds_b.squeeze(0).cpu().float().numpy().astype(np.float16)
            processed_data["text_attention_mask"] = text_mask_b.squeeze(0).cpu().numpy().astype(np.uint8)
            
            mert_list = static_infer_ssl_features(wav_48k_batch, wav_lens_48k_batch, models_dict['mert_model'], models_dict['resampler_mert'], device, 24000, 5)
            mert_feat = mert_list[0] if mert_list and mert_list[0].numel() > 0 else torch.empty(0, models_dict['mert_model'].config.hidden_size, device=device)
            processed_data["mert_ssl_hidden_states"] = mert_feat.cpu().float().numpy().astype(np.float16)

            mhub_list = static_infer_ssl_features(wav_48k_batch, wav_lens_48k_batch, models_dict['hubert_model'], models_dict['resampler_mhubert'], device, 16000, 30)
            mhub_feat = mhub_list[0] if mhub_list and mhub_list[0].numel() > 0 else torch.empty(0, models_dict['hubert_model'].config.hidden_size, device=device)
            processed_data["mhubert_ssl_hidden_states"] = mhub_feat.cpu().float().numpy().astype(np.float16)

            lyrics_raw = item_data.get("norm_lyrics", "[instrumental]")
            lyric_ids_list = tokenize_lyrics_preprocess(lyrics_raw)[:4096]
            lyric_ids = torch.tensor(lyric_ids_list, dtype=torch.long)
            processed_data["norm_lyrics"] = lyrics_raw           # <— add this
            processed_data["lyric_token_ids"] = lyric_ids.cpu().numpy().astype(np.int32)
            processed_data["lyric_masks"] = torch.ones_like(lyric_ids).cpu().numpy().astype(np.uint8)
    except Exception as model_e:
        logger.error(f"SKIPPING item {key} ({audio_filename}) due to model processing error: {model_e}", exc_info=True)
        if device.type == 'cuda': torch.cuda.empty_cache()
        return None

    processed_data.update({
        "keys": key, "original_filename": audio_filename, "original_wav_length": original_wav_length,
        "processed_wav_length": current_wav_len_48k,
        "speaker_embds": torch.zeros(512, dtype=torch.float32).cpu().float().numpy().astype(np.float16)})
    if device.type == 'cuda': torch.cuda.empty_cache()
    return processed_data

def load_all_models_static(checkpoint_dir_str, device_obj):
    ckpt_path = Path(checkpoint_dir_str)
    logger.info(f"Loading ACEStepPipeline from {ckpt_path}...")
    ace_pipe = ACEStepPipeline(checkpoint_dir=str(ckpt_path)) 
    ace_pipe.load_checkpoint(str(ckpt_path)) 
    
    models = {
        'dcae': ace_pipe.music_dcae.to(device_obj).eval(),
        'text_encoder_model': ace_pipe.text_encoder_model.to(device_obj).eval(),
        'text_tokenizer': ace_pipe.text_tokenizer,
        'mert_model': AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=str(ckpt_path)).to(device_obj).eval(),
        'resampler_mert': torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000).to(device_obj),
        'processor_mert': Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=str(ckpt_path)),
        'hubert_model': AutoModel.from_pretrained("utter-project/mHuBERT-147", trust_remote_code=True, cache_dir=str(ckpt_path)).to(device_obj).eval(),
        'resampler_mhubert': torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(device_obj),
        'processor_mhubert': Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147", trust_remote_code=True, cache_dir=str(ckpt_path))
    }
    logger.info("All preprocessing models loaded.")
    return models

def main(args):
    device_str = args.device
    if "cuda" in device_str and not torch.cuda.is_available():
        logger.warning(f"CUDA device {device_str} specified but CUDA not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    max_duration = 240.0
    logger.info(f"Max audio duration for preprocessing: {max_duration}s")

    audio_root = Path(args.audio_root_path).resolve()
    if not audio_root.is_dir(): logger.error(f"Audio root path {audio_root} not found."); return
    logger.info(f"Audio root: {audio_root}")

    models = load_all_models_static(args.checkpoint_dir, device)

    logger.info(f"Loading input HF dataset from: {args.input_dataset_path}")
    try:
        input_ds = load_from_disk(args.input_dataset_path)
    except Exception as e: logger.error(f"Failed to load dataset {args.input_dataset_path}: {e}", exc_info=True); return
    logger.info(f"Input dataset: {len(input_ds)} items.")

    processed_list = [
        data for item in tqdm(input_ds, desc="Preprocessing dataset") 
        if (data := preprocess_item_data_static(item, models, device, audio_root, max_duration)) is not None
    ]

    if not processed_list: logger.error("No items processed successfully. Exiting."); return
    logger.info(f"Successfully processed {len(processed_list)} items.")
    
    try:
        # processed_hf_ds = Dataset.from_list(processed_list)
        processed_hf_ds = build_dataset_in_shards(processed_list, shard_size=25)

    except Exception as e:
        logger.error(f"Error creating Dataset from list: {e}", exc_info=True)
        if processed_list:
            logger.error(f"First item keys: {list(processed_list[0].keys())}")
            for k, v in processed_list[0].items(): logger.error(f"  {k}: type {type(v)}, shape {getattr(v, 'shape', 'N/A')}, dtype {getattr(v, 'dtype', 'N/A')}")
        return

    out_path = Path(args.output_cache_path)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving preprocessed dataset to: {out_path}")
    try:
        processed_hf_ds.save_to_disk(str(out_path))
    except Exception as e: logger.error(f"Error saving dataset: {e}", exc_info=True); return

    logger.info(f"Saved {len(processed_hf_ds)} items. Features: {processed_hf_ds.features}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for Text2Music training.")
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--audio_root_path", type=str, required=True)
    parser.add_argument("--output_cache_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)