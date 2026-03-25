import json

def convert_to_txt(json_file_path, output_file_path):
    """
    Convert JSON data to TXT format with original and translation lines.
    
    Args:
        json_file_path: Path to the input JSON file
        output_file_path: Path to the output TXT file
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Skip the first item which contains the canary key
            if 'big-bench-canary' in item:
                continue
            
            original = item.get('original', '')
            translation = item.get('translation', '')
            
            if original or translation:
                f.write(f"original: {original}\n")
                f.write(f"translation: {translation}\n")
                f.write("\n")  # Add a blank line between entries

# Usage
if __name__ == "__main__":
    # Assuming the JSON data is in a file called 'kalamang_sentences.json'
    convert_to_txt('datasets/kalamang/splits/train_examples.json', 'datasets/kalamang/splits/train_examples.txt')
