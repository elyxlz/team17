import os
import time
from datetime import datetime
from pathlib import Path


def find_recent_files(directory, hours=1):
    """
    Find files created within the specified number of hours.
    Returns a list of files with their creation times for verification.
    """
    current_time = time.time()
    hour_ago = current_time - (hours * 3600)
    recent_files = []

    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            # Get creation time (ctime for UNIX, getctime for Windows)
            creation_time = (
                file_path.stat().st_birthtime
                if hasattr(os.stat_result, "st_birthtime")
                else file_path.stat().st_ctime
            )

            if creation_time > hour_ago:
                creation_datetime = datetime.fromtimestamp(creation_time)
                recent_files.append((str(file_path), creation_datetime))

    return recent_files


def delete_files(files):
    """
    Delete the specified files and return the count of deleted files.
    """
    deleted_count = 0
    for file_path, _ in files:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return deleted_count


def main():
    # Get directory path from user
    directory = input("Enter the directory path: ").strip()

    # Validate directory exists
    if not os.path.isdir(directory):
        print("Error: Directory does not exist!")
        return

    # Find recent files
    recent_files = find_recent_files(directory)

    if not recent_files:
        print("No files found that were created in the last hour.")
        return

    # Display files to be deleted
    print("\nFiles created within the last hour:")
    print("-" * 70)
    for file_path, creation_time in recent_files:
        print(f"Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')} - {file_path}")
    print("-" * 70)
    print(f"\nTotal files to be deleted: {len(recent_files)}")

    # Get confirmation
    confirmation = input(
        "\nAre you sure you want to delete these files? (yes/no): "
    ).lower()

    if confirmation == "yes":
        deleted_count = delete_files(recent_files)
        print(f"\nSuccessfully deleted {deleted_count} files.")
    else:
        print("\nOperation cancelled.")


if __name__ == "__main__":
    main()
