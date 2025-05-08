import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm

# Supported audio file extensions for input (can be expanded)
SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

def find_audio_files(input_dir):
    """Recursively finds all supported audio files in the input directory."""
    audio_files = []
    print(f"Searching for audio files in: {input_dir}")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(Path(root) / file)
    print(f"Found {len(audio_files)} potential audio files.")
    return audio_files

def get_max_duration(audio_files):
    """Determines the maximum duration (in milliseconds) among all audio files."""
    max_duration_ms = 0
    durations = {}  # To store durations of successfully decoded files
    if not audio_files:
        return 0, {}

    print("Scanning for maximum audio duration...")
    for file_path in tqdm(audio_files, desc="Scanning durations"):
        try:
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)
            durations[file_path] = duration_ms  # Store duration for successfully read files
            if duration_ms > max_duration_ms:
                max_duration_ms = duration_ms
        except CouldntDecodeError:
            print(f"Warning: Could not decode {file_path} during duration scan. Skipping for max duration check.")
        except Exception as e:
            print(f"Warning: Error processing {file_path} for duration: {e}. Skipping for max duration check.")
    
    if not durations and max_duration_ms == 0 and audio_files:
        print("Warning: No audio files could be successfully read to determine duration.")

    return max_duration_ms, durations

def process_audio_file(file_path, output_dir, max_duration_ms, input_dir_root):
    """
    Loads an audio file, truncates it if longer than max_duration_ms, loops it if shorter,
    and saves it to the output directory as a .wav file.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        current_duration_ms = len(audio)

        if current_duration_ms >= max_duration_ms:
            # Truncate to max_duration_ms
            print(f"Truncating {file_path} from {current_duration_ms/1000.0:.2f}s to {max_duration_ms/1000.0:.2f}s.")
            padded_audio = audio[:max_duration_ms]  # Truncate to max_duration_ms
        else:
            # Loop the audio to reach at least max_duration_ms
            looped_audio = AudioSegment.empty()  # Start with an empty AudioSegment
            while len(looped_audio) < max_duration_ms:
                looped_audio += audio  # Concatenate the audio repeatedly
            padded_audio = looped_audio[:max_duration_ms]  # Trim to exact length

        # Determine output path, preserving subdirectory structure
        relative_path = file_path.relative_to(input_dir_root)
        output_filename_with_wav_suffix = relative_path.with_suffix('.wav')
        output_file_path = output_dir / output_filename_with_wav_suffix

        # Create output subdirectory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the audio as WAV format
        padded_audio.export(output_file_path, format='wav')
        return True, None
    except CouldntDecodeError:
        error_msg = f"Could not decode {file_path}. Skipping processing."
        print(f"Warning: {error_msg}")
        return False, error_msg
    except FileNotFoundError:
        error_msg = f"Error processing {file_path}: ffmpeg or ffprobe not found. Please ensure they are installed and in your system's PATH."
        print(f"Critical: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error processing and saving {file_path}: {e}. Skipping."
        print(f"Warning: {error_msg}")
        return False, error_msg
    
def main():
    parser = argparse.ArgumentParser(
        description="Standardize the length of audio files in a folder and its subfolders "
                    "by truncating longer files and looping shorter files to match "
                    "the specified or detected maximum duration. All output files are saved in WAV format."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the root folder containing audio files."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where processed WAV audio files will be saved."
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=80.0,
        help="Maximum duration in seconds for all audio files. "
             "Files longer than this will be truncated, and shorter files will be looped to meet this duration. "
             "If set to 0 or negative, the script will auto-detect the longest duration among input files. "
             "Default is 80 seconds."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    max_duration_ms = int(args.max_duration * 1000)  # Convert seconds to milliseconds

    if not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist or is not a directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input folder: {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")

    audio_files = find_audio_files(input_dir)

    if not audio_files:
        print("No supported audio files found in the input folder or its subdirectories.")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    # Determine max duration: use specified value or auto-detect
    if max_duration_ms <= 0:
        print("Max duration set to 0 or negative. Auto-detecting maximum duration from audio files...")
        max_duration_ms, durations_map = get_max_duration(audio_files)
    else:
        print(f"Using specified max duration: {max_duration_ms/1000.0:.2f} seconds.")
        durations_map = {}  # No need to scan durations if max_duration is specified

    if max_duration_ms == 0:
        if audio_files:
            print("Could not determine maximum duration because no audio files could be read successfully.")
            print("Please check warnings above for issues with individual files.")
            print("Ensure ffmpeg is installed for formats like MP3, M4A, etc.")
        return

    print(f"Maximum audio duration: {max_duration_ms/1000.0:.2f} seconds.")

    print("\nProcessing audio files (truncating or padding, outputting as WAV)...")
    processed_count = 0
    failed_count = 0

    # Process all found audio files
    files_to_process = audio_files

    for file_path in tqdm(files_to_process, desc="Processing files"):
        success, error = process_audio_file(file_path, output_dir, max_duration_ms, input_dir)
        if success:
            processed_count += 1
        else:
            failed_count += 1
            if error and "ffmpeg or ffprobe not found" in error:
                print("Critical error related to ffmpeg/ffprobe. Aborting further processing.")
                return

    print(f"\nProcessing complete.")
    print(f"Successfully processed and saved as WAV: {processed_count} files.")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} files (see warnings above).")
    print(f"Processed files saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()