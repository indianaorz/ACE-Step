#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def populate_trackinfo(folder_path_str: str, game_name: str, default_tags: list[str], output_file_str: str):
    """
    Populates a trackinfo.json file with entries from a specified folder (recursing into subfolders).

    Args:
        folder_path_str: Path to the folder containing audio files.
        game_name: Name of the game.
        default_tags: A list of default tags for new entries.
        output_file_str: Path to the output trackinfo.json file.
    """
    input_folder = Path(folder_path_str)
    output_file = Path(output_file_str)

    # Validate input folder
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    if not input_folder.is_dir():
        print(f"Error: Input path '{input_folder}' is not a directory.")
        return

    # Load existing data if output file exists
    all_tracks_data: dict[str, dict] = {}
    if output_file.exists():
        try:
            with output_file.open('r', encoding='utf-8') as f:
                all_tracks_data = json.load(f)
            print(f"Loaded existing data from '{output_file}'.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{output_file}'. Starting fresh.")
        except Exception as e:
            print(f"Warning: Could not read '{output_file}' ({e}). Starting fresh.")

    folder_key_prefix = input_folder.name  # e.g. "Album"
    print(f"\nProcessing files in folder: '{input_folder}' using key prefix: '{folder_key_prefix}'")

    files_processed_count = 0
    new_entries_count = 0
    updated_entries_count = 0

    # Recurse into all subfolders
    for item_path in input_folder.rglob("*"):
        if not item_path.is_file():
            continue

        # Build the JSON key as "FolderName/subdir/.../filename.ext"
        rel_path = item_path.relative_to(input_folder).as_posix()
        track_key = f"{folder_key_prefix}/{rel_path}"

        # Build default entry
        track_entry = {
            "game":        game_name,
            "boss":        None,
            "stage":       "",
            "description": "",
            "tags":        list(default_tags),
            "instruments": []
        }

        if track_key in all_tracks_data:
            updated_entries_count += 1
        else:
            new_entries_count += 1

        all_tracks_data[track_key] = track_entry
        files_processed_count += 1
        print(f"  Processed: {track_key}")

    if files_processed_count == 0:
        print("No files found in the input folder (or subfolders) to process.")
    else:
        print(f"\nProcessed {files_processed_count} file(s).")
        print(f"  Added {new_entries_count} new entries.")
        print(f"  Updated {updated_entries_count} existing entries.")

    # Ensure output directory exists
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_file.parent}': {e}")
        return

    # Write out the JSON
    try:
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(all_tracks_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully wrote track information to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate a trackinfo.json file from a folder of audio tracks (including subfolders).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the audio files."
    )
    parser.add_argument(
        "game",
        type=str,
        help="Name of the game."
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Default tags to add to each track (e.g., vgm instrumental)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trackinfo.json",
        help="Path to the output trackinfo.json file (default: trackinfo.json)."
    )

    args = parser.parse_args()
    print("Starting trackinfo population script...")
    populate_trackinfo(args.folder, args.game, args.tags, args.output)
    print("Script finished.")
