import csv
import sys
import uuid

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

def remove_diacritics(text):
    # Replace Kyrgyz diacritical marks with base letters
    replacements = {
        'ө': 'о',
        'ң': 'н',
        'ү': 'у'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def process_file(input_file, output_file):
    sentences_processed = 0
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()  # Convert to lowercase
            if line and not line.startswith(('title:', 'url:', 'created at:', 'posted by:', 'author:', 'translator:', 'editor:', 'sentences:', '----------------------------------------')):
                # Create source (without diacritics) and target (original) pairs
                source = remove_diacritics(line)
                target = line
                if source != target:  # Only include if there's actually a difference
                    # Generate a UUID for this pair
                    row_id = str(uuid.uuid4())
                    data.append([row_id, source, target])
                    sentences_processed += 1
                    if sentences_processed % 1000 == 0:  # Print progress every 1000 sentences
                        print(f"Processed {sentences_processed} sentences...")
    
    # Write to TSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'source', 'target'])  # Updated header
        writer.writerows(data)
    
    print(f"\nTotal processed: {len(data)} sentence pairs")
    
    # Save examples to a separate file for viewing
    with open('examples.txt', 'w', encoding='utf-8') as f:
        f.write("First few examples:\n\n")
        for i, (row_id, source, target) in enumerate(data[:5]):  # Showing 5 examples now
            f.write(f"{i+1}. ID: {row_id}\n")
            f.write(f"   Source: {source}\n")
            f.write(f"   Target: {target}\n\n")
    
    print("Examples have been saved to examples.txt")

if __name__ == '__main__':
    input_file = 'all_texts.txt'
    output_file = 'dataset.tsv'
    process_file(input_file, output_file)
    print(f"Dataset created successfully! Check {output_file}")
