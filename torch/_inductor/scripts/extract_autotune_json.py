import os
import json

# Path to the cache directory
cache_dir = '/tmp/torchinductor_root/cache'
output_dir = './autotune_ops'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process each file in the cache directory
def process_cache_files():
    # Iterate through each file in the cache directory
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        
        # Skip directories, we only want files
        if os.path.isdir(file_path):
            continue
        
        # Try to open and load the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check if the 'cache' key exists in the JSON
                if 'cache' in data:
                    cache_data = data['cache']
                    
                    # Process each key in the 'cache' section
                    for cache_key, cache_value in cache_data.items():
                        # Define the output file path for this cache entry
                        output_file_path = os.path.join(output_dir, f"{cache_key}.json")
                        
                        # Save the cache entry as a separate JSON file
                        with open(output_file_path, 'w') as output_file:
                            json.dump({cache_key: cache_value}, output_file, indent=4)
                            
                        print(f"Saved cache entry '{cache_key}' to {output_file_path}")
        
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

# Run the process
if __name__ == '__main__':
    process_cache_files()

