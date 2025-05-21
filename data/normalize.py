import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm
import math # For math.isinf

# Supported audio file extensions for input (can be expanded)
SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

# Define thresholds for categorization (in dB of gain applied)
# Gain = Target Peak (-headroom) - Original Peak
GREEN_THRESHOLD = 3.0  # Gain < 3 dB
YELLOW_THRESHOLD = 10.0 # 3 dB <= Gain < 10 dB
# RED is Gain >= 10 dB or original was silent

def find_audio_files(input_dir_path: Path):
    """
    Recursively finds all supported audio files in the input directory.

    Args:
        input_dir_path: The Path object of the directory to search.

    Returns:
        A list of Path objects, each representing an audio file.
    """
    audio_files = []
    print(f"Searching for audio files in: {input_dir_path}")
    for root, _, files in os.walk(input_dir_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(file_path)
    print(f"Found {len(audio_files)} potential audio files.")
    return audio_files

def normalize_audio_file(file_path: Path, headroom: float = 0.1):
    """
    Loads an audio file, normalizes its loudness, and overwrites the original file.
    Also calculates the gain that was applied.

    Args:
        file_path: The Path object of the audio file to process.
        headroom: The desired headroom in dB below full scale for the loudest peak.
                  Defaults to 0.1 dB.

    Returns:
        A tuple (bool, str or None, float or None):
        - True if successful, False otherwise.
        - An error message string if an error occurred, otherwise None.
        - The gain applied in dB if successful, otherwise None.
    """
    try:
        # Determine the original format to save it back in the same format
        original_format = file_path.suffix.lower().lstrip('.')
        if original_format == 'm4a': # pydub uses 'ipod' for m4a
            original_format = 'ipod'
        elif original_format == 'aac':
             pass # format='aac' is fine for export

        # Load the audio file
        audio = AudioSegment.from_file(file_path, format=original_format if original_format in ['aac', 'ipod'] else None)

        original_max_dbfs = audio.max_dBFS

        # Normalize the audio
        normalized_audio = audio.normalize(headroom=headroom)

        # Calculate gain applied
        # Target peak is -headroom dBFS.
        # Gain applied = target_peak - original_peak
        target_peak_dbfs = -abs(headroom) # Ensure headroom is treated as positive value for calculation
        
        if math.isinf(original_max_dbfs) and original_max_dbfs < 0: # Original was silent
            # If original was silent and normalize() results in silent, gain is effectively 0 for no change.
            # However, to reach a target level from silence would be infinite gain.
            # We'll consider this a large gain for categorization if it *becomes* audible,
            # but pydub's normalize leaves silent audio silent.
            # If it's still silent after normalize, normalized_audio.max_dBFS will also be -inf.
            if math.isinf(normalized_audio.max_dBFS) and normalized_audio.max_dBFS < 0:
                 gain_applied = 0.0 # No change to silent audio
            else: # Silent audio became audible (unlikely with pydub.normalize but for robustness)
                 gain_applied = float('inf')
        elif math.isinf(original_max_dbfs) and original_max_dbfs > 0: # Original was +inf (e.g. corrupt)
            gain_applied = float('-inf') # Effectively infinite attenuation
        else:
            gain_applied = target_peak_dbfs - original_max_dbfs


        # Overwrite the original file
        export_params = {}
        if original_format == 'mp3':
            pass # Let pydub use defaults or configure as needed
        
        normalized_audio.export(file_path, format=original_format)
        return True, None, gain_applied

    except CouldntDecodeError:
        msg = f"Could not decode {file_path}. File may be corrupted or unsupported."
        print(f" Warning: {msg}")
        return False, msg, None
    except FileNotFoundError:
        msg = "ffmpeg/ffprobe not found. Please install it and add it to your system's PATH."
        print(f" Critical: {msg}")
        return False, msg, None
    except Exception as e:
        if "No such file or directory" in str(e) and "ffmpeg" in str(e).lower():
             msg = f"ffmpeg/ffprobe not found or error during its execution for {file_path}: {e}"
        else:
            msg = f"An unexpected error occurred while processing {file_path}: {e}"
        print(f" Error: {msg}")
        return False, msg, None

def main():
    parser = argparse.ArgumentParser(
        description="Normalizes the loudness of all supported audio files within a specified folder. "
                    "WARNING: This script overwrites the original files."
    )
    parser.add_argument("input_folder", type=str,
                        help="The root folder containing audio files to normalize.")
    parser.add_argument("--headroom", type=float, default=0.1,
                        help="Target headroom in dBFS for normalization (e.g., 0.1). "
                             "This is the level below full scale for the loudest peak. Default is 0.1 dB.")

    args = parser.parse_args()

    input_dir = Path(args.input_folder).resolve()

    if not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' is not a valid directory.")
        return

    print("--- WARNING ---")
    print("This script will OVERWRITE original audio files in the specified folder and its subfolders.")
    print(f"Target folder: {input_dir}")
    print(f"Normalization headroom: {args.headroom} dBFS")
    print("Please ensure you have backups if you don't want to lose original versions.")

    if input("Proceed with normalization? (yes/no): ").strip().lower() != 'yes':
        print("Normalization cancelled by the user.")
        return

    all_audio_files = find_audio_files(input_dir)
    if not all_audio_files:
        print("No supported audio files found in the input folder.")
        return

    print(f"\nStarting normalization for {len(all_audio_files)} files...")

    processed_count = 0
    failed_count = 0
    category_counts = {"green": 0, "yellow": 0, "red": 0}

    for file_path in tqdm(all_audio_files, desc="Normalizing files", total=len(all_audio_files)):
        success, error_message, gain_applied = normalize_audio_file(file_path, headroom=args.headroom)
        if success:
            processed_count += 1
            if gain_applied is not None:
                if math.isinf(gain_applied) and gain_applied > 0: # Was silent, became audible (or huge gain)
                    category_counts["red"] += 1
                elif gain_applied >= YELLOW_THRESHOLD: # Significant amplification
                    category_counts["red"] += 1
                elif gain_applied >= GREEN_THRESHOLD: # Moderate amplification
                    category_counts["yellow"] += 1
                else: # Minor amplification, no change, or attenuation
                    category_counts["green"] += 1
            else: # Should not happen if success is True, but as a fallback
                category_counts["green"] += 1 # Assume minor if gain unknown but successful
        else:
            failed_count += 1
            if error_message and "ffmpeg/ffprobe not found" in error_message:
                print("\nCritical ffmpeg/ffprobe error encountered. Aborting further processing.")
                break 

    print(f"\n--- Normalization Complete ---")
    print(f"  Successfully processed and overwritten: {processed_count} files.")
    print(f"  Failed to process: {failed_count} files.")

    if processed_count > 0:
        print("\n--- Normalization Impact ---")
        print(f"  🟢 Green : {category_counts['green']:<4} files (already well-normalized or minor adjustments needed).")
        print(f"  🟡 Yellow: {category_counts['yellow']:<4} files (moderately quieter, noticeable normalization applied).")
        print(f"  🔴 Red   : {category_counts['red']:<4} files (significantly quieter, major normalization applied).")
    
    if failed_count > 0:
        print("\nPlease check the console output for specific error messages for failed files.")

if __name__ == "__main__":
    main()
