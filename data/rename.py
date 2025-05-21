import os
import argparse

def rename_files_after_last_hyphen(folder_path):
    """
    Renames files in the specified folder.
    The new name is the part of the original filename after the last hyphen.
    File extensions are preserved.

    Args:
        folder_path (str): The path to the folder containing the files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist or is not a directory.")
        return

    print(f"Processing files in folder: '{folder_path}'\n")
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for filename in os.listdir(folder_path):
        original_full_path = os.path.join(folder_path, filename)

        # Ensure it's a file and not a directory
        if os.path.isfile(original_full_path):
            base_name, extension = os.path.splitext(filename)

            # Find the last hyphen in the base name
            last_hyphen_index = base_name.rfind('-')

            if last_hyphen_index != -1:
                # Extract the part after the last hyphen
                new_base_name = base_name[last_hyphen_index + 1:]

                if new_base_name:  # Ensure the new base name is not empty
                    new_filename = new_base_name + extension
                    #trim
                    new_filename = new_filename.strip()
                    new_full_path = os.path.join(folder_path, new_filename)

                    # Avoid renaming if new name is the same as old name
                    if original_full_path == new_full_path:
                        print(f"Skipping '{filename}': New name is identical to the old name.")
                        skipped_count += 1
                        continue

                    # Check if a file with the new name already exists
                    if os.path.exists(new_full_path):
                        print(f"Skipping '{filename}': Target file '{new_filename}' already exists.")
                        skipped_count += 1
                        continue

                    try:
                        os.rename(original_full_path, new_full_path)
                        print(f"Renamed: '{filename}' -> '{new_filename}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Error renaming '{filename}' to '{new_filename}': {e}")
                        error_count += 1
                else:
                    print(f"Skipping '{filename}': The part after the last hyphen is empty.")
                    skipped_count += 1
            else:
                print(f"Skipping '{filename}': No hyphen found in the base name.")
                skipped_count += 1
        else:
            # This entry is a directory or something else, not a regular file.
            # You can add a message here if you want to log skipped directories.
            # print(f"Skipping non-file entry: '{filename}'")
            pass

    print(f"\n--- Summary ---")
    print(f"Files successfully renamed: {renamed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename files in a folder to the section after the last hyphen in their names."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="The path to the folder containing files to be renamed."
    )
    args = parser.parse_args()
    
    rename_files_after_last_hyphen(args.folder)