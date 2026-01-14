import os
import csv

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, "TSB-AD-U")
OUTPUT_NAME = "kept_csvs.csv"
OUTPUT_CSV = os.path.join(FOLDER_PATH, OUTPUT_NAME)
MAX_ROWS = 10000

kept_files = []

for filename in os.listdir(FOLDER_PATH):
    if not filename.lower().endswith(".csv"):
        continue

    # DO NOT PROCESS THE OUTPUT FILE
    if filename == OUTPUT_NAME:
        continue

    file_path = os.path.join(FOLDER_PATH, filename)

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        row_count = sum(1 for row in reader if any(cell.strip() for cell in row))

    if row_count <= MAX_ROWS:
        kept_files.append(filename)

# WRITE OUTPUT
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["csv_name"])
    for name in kept_files:
        writer.writerow([name])

print("DONE")
print("Files kept:", len(kept_files))
