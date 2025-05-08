#!/usr/bin/env python3
# build_hf_dataset.py
# ---------------------------------------------------------------------------
#  Creates a Hugging Face dataset for a dataset
#
#  Required:  pip install datasets  (and peft if you plan LoRA fine-tuning)
# ---------------------------------------------------------------------------

import hashlib
import uuid
import json # Added for JSON loading
from pathlib import Path
from datetime import datetime
from datasets import Dataset

# ═════════════════════════ TRACK METADATA ════════════════════════════════════
#  TRACK_INFO will loaded from datasetlabeller/trackinfo.json
# ----------------------------------------------------------------------------
TRACK_INFO = {}


def load_track_info(json_path: Path) -> dict:
    """Loads track information from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded TRACK_INFO from {json_path}")
        return data
    except FileNotFoundError:
        print(f"❌ ERROR: TRACK_INFO file not found at {json_path}")
        print("Please ensure 'datasetlabeller/trackinfo.json' exists.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from {json_path}")
        print("Please ensure 'datasetlabeller/trackinfo.json' is a valid JSON file.")
        return {}
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading {json_path}: {e}")
        return {}


def make_key(wav_relative_path: Path) -> str:
    """
    Creates a unique key for a track.
    Using a simple hash of the relative path for reproducibility.
    A more robust approach might involve hashing file content if needed,
    but for dataset linking, relative path is usually sufficient.
    """
    # Using SHA256 hash of the relative path string
    hasher = hashlib.sha256()
    hasher.update(wav_relative_path.as_posix().encode('utf-8'))
    # Return the first 16 characters of the hex digest for a shorter key
    return hasher.hexdigest()[:16]


def build_recaptions(tags_list):
    """
    Builds a recaption string from a list of tags.
    Currently joins all tags.
    """
    if tags_list:
        return {"default": ", ".join(tags_list)}
    return {"default": ""}


def main():
    global TRACK_INFO # Declare TRACK_INFO as global to modify it

    root = Path(".").resolve()  # Assumes script is run from a dir containing MMX, MMX2 etc.
    # Load TRACK_INFO from JSON file
    track_info_path = root / "trackinfo.json"
    
    TRACK_INFO = load_track_info(track_info_path)

    if not TRACK_INFO:
        print("❌ Exiting script as TRACK_INFO could not be loaded.")
        return

    rows = []

    # 1. Get expected files from TRACK_INFO (relative paths)
    expected_files_rel_set = set(TRACK_INFO.keys())

    # 2. Get actual .wav files from the directory structure (relative paths)
    actual_wav_files_abs_paths = list(root.rglob("*.wav"))
    actual_wav_files_rel_set = set()
    actual_files_map_rel_to_abs = {}

    for abs_path in actual_wav_files_abs_paths:
        try:
            # Ensure consistent path separators (POSIX) for keys
            rel_path_posix = abs_path.relative_to(root).as_posix()
            actual_wav_files_rel_set.add(rel_path_posix)
            actual_files_map_rel_to_abs[rel_path_posix] = abs_path
        except ValueError:
            print(f"[INFO] Found a .wav file outside the main directory structure: {abs_path}")

    # 3. Compare and log
    missing_files = expected_files_rel_set - actual_wav_files_rel_set
    if missing_files:
        print("\n⚠️ WARNING: The following files defined in TRACK_INFO are MISSING from disk:")
        for f in sorted(list(missing_files)):
            print(f"  - {f}")
    else:
        print("\n✅ All files defined in TRACK_INFO appear to be present on disk.")

    extra_files = actual_wav_files_rel_set - expected_files_rel_set
    if extra_files:
        print("\n⚠️ WARNING: The following .wav files on disk are NOT DEFINED in TRACK_INFO:")
        for f in sorted(list(extra_files)):
            print(f"  - {f}")
    else:
        print("\n✅ No undefined .wav files found on disk (in tracked directories).")

    print("\nProcessing files...\n")

    processed_files_count = 0
    # Iterate through TRACK_INFO to maintain order and metadata association
    for rel_path_str, meta_from_dict in TRACK_INFO.items():
        if rel_path_str not in actual_wav_files_rel_set:
            continue

        wav_abs_path = actual_files_map_rel_to_abs[rel_path_str]
        wav_relative_path_for_key = wav_abs_path.relative_to(root) # Path object for make_key

        current_tags = list(meta_from_dict.get("tags", [])).copy()
        description = meta_from_dict.get("description")
        album = meta_from_dict.get("album")
        trackname = meta_from_dict.get("trackname") 
        instruments = meta_from_dict.get("instruments", []) # Default to empty list

        # Prepend specific metadata as tags if they exist
        prefix_tags = []
        if album:
            prefix_tags.append(str(album)) # Ensure game is a string
        if trackname:
            prefix_tags.append(str(trackname)) # Ensure boss is a string
        
        for instrument in instruments:
            if str(instrument) not in current_tags and str(instrument) not in prefix_tags: # Ensure instrument is string and not duplicate
                prefix_tags.append(str(instrument))
        
        # Add description to tags if it exists and isn't already a tag
        if description and description not in current_tags and description not in prefix_tags:
            # Decide if description should be a prefix or suffix, here it's appended to specific metadata
            prefix_tags.append(description)


        final_tags = prefix_tags + current_tags
        # Remove potential duplicates that might arise if description was already in tags
        final_tags = sorted(list(set(final_tags)), key=lambda x: (prefix_tags + current_tags).index(x))


        rows.append({
            "keys": make_key(wav_relative_path_for_key),
            "filename": rel_path_str,
            "tags": final_tags,
            "speaker_emb_path": "",  # Placeholder
            "norm_lyrics": "[instrumental]",  # Assuming all are instrumental
            "recaption": build_recaptions(final_tags),
        })
        processed_files_count += 1
        print(f"[ADDED] {rel_path_str} with tags: {final_tags}")

    if not rows:
        print("\n❌ No data processed. Dataset will be empty. Check file paths, TRACK_INFO, and logs.")
        return

    ds = Dataset.from_list(rows)
    out_dir = root / "dataset" # Ensure out_dir is relative to root if desired
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ds.save_to_disk(out_dir)
        print(f"\n✅ Saved {len(ds)} examples to {out_dir}.")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save dataset to {out_dir}: {e}")
        return

    print(f"ℹ️ Processed {processed_files_count} files out of {len(TRACK_INFO)} total tracks defined in TRACK_INFO.")
    if missing_files:
        print(f"ℹ️ {len(missing_files)} defined tracks were skipped due to missing audio files.")


if __name__ == "__main__":
    main()