import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm

# Supported audio file extensions for input (can be expanded)
SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

def find_audio_files(input_dir_path: Path):
    """Recursively finds all supported audio files in the input directory."""
    audio_files = []
    print(f"Searching for audio files in: {input_dir_path}")
    for root, _, files in os.walk(input_dir_path):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(Path(root) / file)
    print(f"Found {len(audio_files)} potential audio files.")
    return audio_files

def process_audio_file(file_path: Path, output_dir_for_bucket: Path,
                       target_bucket_duration_ms: int, input_processing_root: Path):
    """
    Loads an audio file, pads it with silence or truncates it to fit
    the target_bucket_duration_ms, and saves it as WAV under output_dir_for_bucket.
    The audio is placed at the beginning of the bucket.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        original_duration_ms = len(audio)

        if original_duration_ms > target_bucket_duration_ms:
            # Truncate the audio
            out_audio = audio[:target_bucket_duration_ms]
        else:
            # Pad with silence
            silence_needed_ms = target_bucket_duration_ms - original_duration_ms
            if silence_needed_ms > 0:
                padding = AudioSegment.silent(duration=silence_needed_ms)
                out_audio = audio + padding
            else:  # Exactly matches bucket duration
                out_audio = audio

        # Preserve subfolder structure relative to the initial input_folder
        try:
            # input_processing_root is the resolved path of the initial input_folder
            # file_path is the resolved path of the current audio file
            relative_path = file_path.relative_to(input_processing_root)
        except ValueError:
            # This might happen if symlinks resolve to paths not strictly under input_processing_root
            # Fallback to just using the filename
            print(f" Warning: Could not determine relative path for {file_path} from {input_processing_root}. Using filename only.")
            relative_path = Path(file_path.name)

        out_file_path = (output_dir_for_bucket / relative_path).with_suffix('.wav')
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        out_audio.export(out_file_path, format='wav')
        return True, None

    except CouldntDecodeError:
        msg = f"Could not decode {file_path}"
        print(f" Warning: {msg}")
        return False, msg
    except FileNotFoundError:  # Typically for ffmpeg/ffprobe
        msg = "ffmpeg/ffprobe not found—install and add to PATH."
        print(f" Critical: {msg}")
        return False, msg
    except Exception as e:
        msg = f"Error processing {file_path}: {e}"
        print(f" Warning: {msg}")
        return False, msg

def main():
    parser = argparse.ArgumentParser(
        description="Processes audio files into duration buckets. "
                    "Files are padded with silence or truncated to fit the chosen bucket. "
                    "Each file is placed into the smallest possible bucket it fits into, "
                    "or truncated to the largest bucket if its duration exceeds all bucket sizes."
    )
    parser.add_argument("input_folder", type=str,
                        help="Root folder of source audio files.")
    parser.add_argument("output_folder", type=str,
                        help="Base folder where to save processed WAV files. "
                             "Subfolders will be created for each bucket duration.")
    parser.add_argument("--buckets", type=float, nargs="+", default=[15.0, 30.0, 60.0, 120.0],
                        help="List of target duration buckets in seconds (e.g., 15 30.5 60). "
                             "Audio files will be processed to fit one of these durations. "
                             "Defaults to 15, 30, 60, and 120 seconds.")
    args = parser.parse_args()

    input_dir = Path(args.input_folder).resolve() # Resolve to absolute path for robust relative path calculation
    output_base_dir = Path(args.output_folder)

    if not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' is not a valid directory.")
        return

    # Ensure output base directory exists (though subdirectories are created later)
    # output_base_dir.mkdir(parents=True, exist_ok=True) # Optional: process_audio_file handles subdirs

    all_audio_files = find_audio_files(input_dir)
    if not all_audio_files:
        print("No supported audio files found in the input folder.")
        return

    # Convert bucket durations from seconds to milliseconds and sort them
    # Use set to ensure unique bucket values, then sort
    unique_sorted_bucket_durations_s = sorted(list(set(b for b in args.buckets if b > 0)))

    if not unique_sorted_bucket_durations_s:
        print("Error: No valid (positive) bucket durations specified. Please provide positive values for --buckets.")
        return

    bucket_durations_ms = [int(s * 1000) for s in unique_sorted_bucket_durations_s]
    largest_bucket_ms = bucket_durations_ms[-1] # The last one after sorting

    print(f"\nWill process {len(all_audio_files)} audio files.")
    print("Target bucket durations (s):", ", ".join(str(s) for s in unique_sorted_bucket_durations_s))

    processed_count = 0
    failed_count = 0

    for file_path in tqdm(all_audio_files, desc="Processing files", total=len(all_audio_files)):
        try:
            # Momentarily load audio to check its original duration
            audio_segment_for_duration_check = AudioSegment.from_file(file_path)
            original_duration_ms = len(audio_segment_for_duration_check)
        except CouldntDecodeError:
            print(f" Warning: Could not decode {file_path} to check its duration. Skipping this file.")
            failed_count += 1
            continue
        except FileNotFoundError:
            print(f" Critical: ffmpeg/ffprobe not found. Cannot process {file_path}. Aborting.")
            return # Critical error, abort.
        except Exception as e:
            print(f" Warning: Error reading duration for {file_path}: {e}. Skipping this file.")
            failed_count += 1
            continue

        # Determine the appropriate bucket for the current audio file
        chosen_bucket_ms = largest_bucket_ms  # Default to the largest bucket
        # Iterate through sorted buckets to find the smallest one that fits
        for bucket_ms in bucket_durations_ms:
            if original_duration_ms <= bucket_ms:
                chosen_bucket_ms = bucket_ms
                break
        # If original_duration_ms > largest_bucket_ms, it remains chosen_bucket_ms = largest_bucket_ms (for truncation)

        chosen_bucket_s = chosen_bucket_ms / 1000.0
        # Create a folder name like "15s" or "30.5s"
        bucket_folder_name = f"{int(chosen_bucket_s) if chosen_bucket_s == int(chosen_bucket_s) else chosen_bucket_s}s"
        output_dir_for_this_file_bucket = output_base_dir / bucket_folder_name

        success, error_message = process_audio_file(file_path, output_dir_for_this_file_bucket,
                                                    chosen_bucket_ms, input_dir)
        if success:
            processed_count += 1
        else:
            failed_count += 1
            # If a critical ffmpeg error occurred during processing, abort all.
            if error_message and "ffmpeg/ffprobe not found" in error_message:
                print("Aborting due to critical ffmpeg/ffprobe error encountered during file processing.")
                return

    print(f"\nProcessing complete.")
    print(f"  Successfully processed: {processed_count} files.")
    print(f"  Failed to process: {failed_count} files.")
    print("Output files are saved under respective bucket subfolders in:")
    print(f"  {output_base_dir.resolve()}")

if __name__ == "__main__":
    main()