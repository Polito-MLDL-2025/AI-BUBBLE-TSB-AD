import csv
import io
import os
import zipfile

import requests

# General settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_LIST_DIR = os.path.join(BASE_DIR, "Datasets/File_List")
MAX_ROWS = 10000
MAX_COLS = 100

# URLs for datasets
DATASET_URLS = {
    'U': 'https://www.thedatum.org/datasets/TSB-AD-U.zip',
    'M': 'https://www.thedatum.org/datasets/TSB-AD-M.zip'
}

def process_type(ts_type):
    folder_path = os.path.join(BASE_DIR, f"Datasets/TSB-AD-{ts_type}")
    
    # Check if dataset is present (at least more than one CSV file)
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        dataset_present = len(csv_files) > 1
    else:
        dataset_present = False
    
    if not dataset_present:
        print(f"Downloading TSB-AD-{ts_type} dataset...")
        url = DATASET_URLS[ts_type]
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(folder_path, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(folder_path)
        
        # Handle nested folder structure: if a folder with the same name was created, flatten it
        nested_folder = os.path.join(folder_path, f"TSB-AD-{ts_type}")
        if os.path.isdir(nested_folder):
            for filename in os.listdir(nested_folder):
                src = os.path.join(nested_folder, filename)
                dst = os.path.join(folder_path, filename)
                if os.path.isfile(src):
                    os.rename(src, dst)
            os.rmdir(nested_folder)
        
        print(f"Downloaded and extracted TSB-AD-{ts_type}")
    
    output_name = f"TSB-AD-{ts_type}-filtered.csv"
    output_csv = os.path.join(FILE_LIST_DIR, output_name)

    kept_files = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".csv"):
            print(f"Skipping non-CSV file {filename}")
            continue

        if filename == output_name:
            print(f"Skipping output file {filename}")
            continue

        file_path = os.path.join(folder_path, filename)

        # Filter all files based on row and column count
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            num_columns = len(header) if header else 0
            row_count = sum(1 for row in reader if any(cell.strip() for cell in row))

        if row_count <= MAX_ROWS and num_columns <= MAX_COLS:
            kept_files.append(filename)

    # Write kept files to output CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name"])
        for name in sorted(kept_files):
            writer.writerow([name])

    print(f"DONE for {ts_type}")
    print(f"Files kept for {ts_type}:", len(kept_files))

    # =============================

    # Load tuning files
    tuning_files = set()
    tuning_path = os.path.join(FILE_LIST_DIR, f"TSB-AD-{ts_type}-Tuning.csv")
    with open(tuning_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:  # ensure not empty
                tuning_files.add(row[0])

    # Load eva files
    eva_files = set()
    eva_path = os.path.join(FILE_LIST_DIR, f"TSB-AD-{ts_type}-Eva.csv")
    with open(eva_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:  # ensure not empty
                eva_files.add(row[0])

    # Filter kept_files
    tuning_kept = [f for f in kept_files if f in tuning_files]
    eva_kept = [f for f in kept_files if f in eva_files]

    # Write tuning filtered
    tuning_output = os.path.join(FILE_LIST_DIR, f"TSB-AD-{ts_type}-Tuning-filtered.csv")
    with open(tuning_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name"])
        for name in sorted(tuning_kept):
            writer.writerow([name])

    # Write eva filtered
    eva_output = os.path.join(FILE_LIST_DIR, f"TSB-AD-{ts_type}-Eva-filtered.csv")
    with open(eva_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name"])
        for name in sorted(eva_kept):
            writer.writerow([name])

    print(f"Tuning files kept for {ts_type}:", len(tuning_kept))
    print(f"Eva files kept for {ts_type}:", len(eva_kept))

if __name__ == "__main__":

    # Process both types
    process_type('U')
    process_type('M')

