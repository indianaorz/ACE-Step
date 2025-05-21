#!/usr/bin/env python3
# build_hf_dataset_from_filenames_with_duration.py
# ---------------------------------------------------------------------------
#  Creates a Hugging Face dataset from WAV files in a directory structure
#  with duration subfolders (e.g., root/20s, root/40s),
#  using filenames to generate tags (foldername and filename only).
#
#  Required:  pip install datasets
# ---------------------------------------------------------------------------

from pathlib import Path
from datasets import Dataset
import argparse


def make_key(full_relative_wav_path: Path) -> str:
    """
    Generate a unique key per file based on its full relative path
    from the root directory (including the duration folder).
    """
    # Replace path separators and dots to make a clean key
    return f"key_{full_relative_wav_path.as_posix().replace('/', '_').replace('.', '_')}"


def build_recaptions(tags_list):
    """
    Builds a recaption dict from a list of tags.
    Currently just joins them.
    """
    return {"default": ", ".join(tags_list)} if tags_list else {"default": ""}


def main():
    p = argparse.ArgumentParser(
        description="Build HF dataset from WAV files in duration subfolders, using filenames as tags."
    )
    p.add_argument(
        "root_dir",
        help="Root dir containing duration subfolders (e.g., '20s', '40s', ...), which in turn contain WAV files."
    )
    p.add_argument(
        "--output_dir",
        default="dataset",
        help="Where to save the HF dataset (default: ./duration_filename_dataset)"
    )
    args = p.parse_args()

    root = Path(args.root_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"Error: Root directory '{root}' does not exist or is not a directory.")
        return

    # 1. Discover all duration folders (assumed to be direct subdirectories of root)
    duration_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    if not duration_folders:
        print(f"Error: No subdirectories (potential duration folders) found under '{root}'.")
        print("Expected structure example: root_dir/20s/audio.wav, root_dir/40s/game_subfolder/audio.wav, etc.")
        return
    
    print(f"Found potential duration folders: {[df.name for df in duration_folders]}")

    rows = []
    total_wav_files_found = 0

    # 2. Scan every WAV in each duration folder
    for duration_dir_path in duration_folders:
        duration_label = duration_dir_path.name
        wav_files_in_duration_folder = 0
        print(f"\nScanning for .wav files in duration folder: '{duration_label}'...")

        # Recursively find all .wav files within the current duration_dir_path
        for wav_path in duration_dir_path.rglob("*.wav"):
            total_wav_files_found += 1
            wav_files_in_duration_folder += 1

            filename_stem = wav_path.stem
            # Path relative to the root includes duration folder: e.g., "20s/Album/Actual Filename.wav"
            full_relative_path_from_root = wav_path.relative_to(root)

            # Create tags: folder name and filename without extension
            #get parent folder name
            parent_folder_name = full_relative_path_from_root.parent.name
            #get parent of parent folder name
            parent_parent_folder_name = full_relative_path_from_root.parent.parent.name
            tags = [parent_parent_folder_name, parent_folder_name, filename_stem]
            print(f"  Found .wav file: '{full_relative_path_from_root}' with tags: {tags}")

            rows.append({
                "keys":             make_key(full_relative_path_from_root),
                "filename":         full_relative_path_from_root.as_posix(),
                "duration":         duration_label,  # e.g., "20s"
                "tags":             tags,
                "speaker_emb_path": "",
                "norm_lyrics":      "[instrumental]",  # Assuming all are instrumental
                "recaption":        build_recaptions(tags),
            })
            if wav_files_in_duration_folder % 100 == 0:  # Log progress every 100 files per folder
                print(f"  Processed {wav_files_in_duration_folder} files in '{duration_label}'...")
        
        if wav_files_in_duration_folder == 0:
            print(f"  No .wav files found in '{duration_label}'.")
        else:
            print(f"  Found {wav_files_in_duration_folder} .wav files in '{duration_label}'.")

    if not rows:
        print(f"\n❌ No .wav files found across all scanned subdirectories in '{root}'. Nothing to save.")
        return

    # 3. Save to disk
    ds = Dataset.from_list(rows)
    ds.save_to_disk(out_dir)
    print(f"\n✅ Processed a total of {total_wav_files_found} .wav files across {len(duration_folders)} scanned duration folder(s).")
    print(f"✅ Wrote {len(ds)} examples to '{out_dir}'.")

if __name__ == "__main__":
    main()
