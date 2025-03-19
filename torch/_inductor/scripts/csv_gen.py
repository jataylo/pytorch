import os
import json
import csv
from collections import defaultdict

def extract_highest_entries(data):
    extracted_data = []
    print("Extracting highest entries...")
    for outer_key, inner_dict in data.items():
        print(f"Processing top-level key: {outer_key}")
        for input_shape, values in inner_dict.items():
            print(f"Processing input shape: {input_shape}")
            input_shape_cleaned = input_shape.replace(",", "")  # Remove commas
            highest = values.get("highest", {})
            if not highest:
                print(f"No 'highest' values found for {input_shape}")
            for key, timing in highest.items():
                if key.startswith("name-"):
                    parts = key.split(":")
                    entry_dict = {"input": input_shape_cleaned}
                    for part in parts:
                        if "-" in part:
                            k, v = part.split("-", 1)
                            entry_dict[k] = v
                    entry_dict["timing"] = timing
                    extracted_data.append(entry_dict)
    return extracted_data

def get_all_dynamic_keys(data_list):
    keys = set()
    for entry in data_list:
        keys.update(entry.keys())
    return ["input"] + sorted(keys - {"input", "timing", "name", "module_cache_key"}) + ["timing"]

def remove_unwanted_columns(entries):
    for entry in entries:
        entry.pop("name", None)
        entry.pop("module_cache_key", None)
    return entries

def process_json_files(directory):
    print(f"Reading JSON files from: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            output_csv = os.path.join(directory, filename.replace(".json", ".csv"))
            print(f"Processing file: {filename}")
            all_entries = []
            
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    print(f"Loaded JSON successfully: {filename}")
                    all_entries.extend(extract_highest_entries(data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {filename}: {e}")
                    continue
            
            if all_entries:
                all_entries = remove_unwanted_columns(all_entries)
                fieldnames = get_all_dynamic_keys(all_entries)
                print(f"Writing CSV: {output_csv} with fields: {fieldnames}")
                
                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_entries)
                
                print(f"CSV file generated: {output_csv}")
            else:
                print(f"No valid entries found in {filename}, skipping CSV generation.")

if __name__ == "__main__":
    process_json_files("./autotune_ops/")

