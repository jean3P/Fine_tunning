import csv
import json
import random
from pathlib import Path


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_json_files(input_csv, output_dir):
    # Read the CSV file
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [
            {'ocr': row['Text with OCR Error'], 'true_label': row['Corrected Text']}
            for row in csv_reader
        ]

    # Split the data
    train_data, val_data, test_data = split_data(data)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the data to JSON files
    with open(Path(output_dir) / 'train.json', 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)
    with open(Path(output_dir) / 'validation.json', 'w', encoding='utf-8') as val_file:
        json.dump(val_data, val_file, ensure_ascii=False, indent=4)
    with open(Path(output_dir) / 'test.json', 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)


# Example usage
input_csv = 'letters_of_George_Whitfield_likely_v3.csv'
output_dir = 'rvl-cdip-ocr'
create_json_files(input_csv, output_dir)
