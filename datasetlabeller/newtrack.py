import argparse
import json
import os
from pathlib import Path

def populate_trackinfo(folder_path_str: str, game_name: str, default_tags: list[str], output_file_str: str):
    """
    Populates a trackinfo.json file with entries from a specified folder.

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

    all_tracks_data = {}

    # Load existing data if output file exists
    if output_file.exists():
        try:
            with output_file.open('r', encoding='utf-8') as f:
                all_tracks_data = json.load(f)
            print(f"Loaded existing data from '{output_file}'.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{output_file}'. Starting with an empty structure.")
        except Exception as e:
            print(f"Warning: Could not read '{output_file}' ({e}). Starting with an empty structure.")

    folder_key_prefix = input_folder.name  # Get the name of the directory itself (e.g., "Album")
    
    print(f"\nProcessing files in folder: '{input_folder}' using key prefix: '{folder_key_prefix}'")
    files_processed_count = 0
    new_entries_count = 0
    updated_entries_count = 0

    for item_path in input_folder.iterdir():
        if item_path.is_file():
            filename = item_path.name
            # Construct the key as "FolderName/filename.ext"
            track_key = f"{folder_key_prefix}/{filename}"

            # Create the default entry structure
            track_entry = {
                "game": game_name,
                "boss": None,  # Will be 'null' in JSON
                "stage": "",
                "description": "",
                "tags": default_tags if default_tags else [],
                "instruments": []
            }

            if track_key not in all_tracks_data:
                new_entries_count += 1
            else:
                updated_entries_count +=1
                
            all_tracks_data[track_key] = track_entry
            files_processed_count += 1
            print(f"  Processed: {track_key}")

    if files_processed_count == 0:
        print("No files found in the input folder to process.")
    else:
        print(f"\nProcessed {files_processed_count} file(s).")
        print(f"Added {new_entries_count} new entries.")
        print(f"Updated {updated_entries_count} existing entries (based on file presence).")


    # Ensure output directory exists
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_file.parent}': {e}")
        return

    # Write data to JSON file
    try:
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(all_tracks_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully wrote track information to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate a trackinfo.json file from a folder of audio tracks.",
        formatter_class=argparse.RawTextHelpFormatter # To allow newlines in help text
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
        nargs='*',  # 0 or more arguments
        default=[],
        help="Default tags to add to each track (e.g., vgm instrumental)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trackinfo.json",
        help="Path to the output trackinfo.json file (default: trackinfo.json in current directory)."
    )

    args = parser.parse_args()

    print("Starting trackinfo population script...")
    populate_trackinfo(args.folder, args.game, args.tags, args.output)
    print("Script finished.")
