import json
import os
import csv
from scipy.stats import pearsonr, kendalltau, spearmanr

# Read JSONL file and parse data
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue  # Skip lines that cannot be parsed
    return data

# Read regular JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # If data is in "results" field, extract it
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            # If it's already a list, return it directly
            elif isinstance(data, list):
                return data
            else:
                # If it's another format, try to process as a single object
                return [data]
        except json.JSONDecodeError:
            # Try reading as JSONL
            f.seek(0)
            return read_jsonl(file_path)

# Calculate correlation metrics
def calculate_correlations(data):
    # Extract true values and predicted values
    # First check if 'score' field exists, if not try using 'label' field
    scores = []
    results = []
    
    for item in data:
        if 'score' in item and item['score'] is not None:
            score = item['score']
        elif 'label' in item and item['label'] is not None:
            score = item['label']
        else:
            continue  # Skip data without scores
            
        if 'result' in item and item['result'] is not None:
            result = item['result']
        else:
            continue  # Skip data without results
            
        # Ensure scores are numerical
        try:
            score = float(score)
            result = float(result)
            scores.append(score)
            results.append(result)
        except (ValueError, TypeError):
            continue
    
    if not scores or not results:
        return {
            "Pearson Correlation": None,
            "Kendall's Tau": None,
            "Spearman Correlation": None,
            "Sample Size": 0
        }
    
    # Calculate correlations
    pearson_corr, _ = pearsonr(scores, results)
    kendall_corr, _ = kendalltau(scores, results)
    spearman_corr, _ = spearmanr(scores, results)

    return {
        "Pearson Correlation": pearson_corr,
        "Kendall's Tau": kendall_corr,
        "Spearman Correlation": spearman_corr,
        "Sample Size": len(scores)
    }

# Main function
def main():
    # Set parameters
    results_dir = input("Please enter the evaluation results directory path: ").strip()
    output_file = input("Please enter the CSV output file path (default is 'correlation_results.csv'): ").strip()
    if not output_file:
        output_file = "correlation_results.csv"
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all JSON and JSONL files in the directory
    model_files = {}
    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        if os.path.isfile(filepath) and (filename.endswith('.json') or filename.endswith('.jsonl')):
            # Extract model name from filename (remove extension)
            model_name = os.path.splitext(filename)[0]
            model_files[model_name] = filepath
    
    if not model_files:
        print(f"No JSON or JSONL files found in directory {results_dir}")
        return
    
    # Read all files and calculate correlations
    results = {}
    
    for model, file_path in model_files.items():
        print(f"Processing {model} model results...")
        
        # Choose reading method based on file extension
        if file_path.endswith('.jsonl'):
            data = read_jsonl(file_path)
        else:
            data = read_json(file_path)
        
        if not data:
            print(f"  Warning: No valid data found in {file_path}")
            continue
        
        results[model] = calculate_correlations(data)
        
        print(f"  Found {results[model]['Sample Size']} valid evaluation items")
    
    # Save results to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model', 'Sample Size', 'Pearson', 'Kendall\'s Tau', 'Spearman']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model, corr in results.items():
            writer.writerow({
                'Model': model,
                'Sample Size': corr['Sample Size'],
                'Pearson': corr['Pearson Correlation'] if corr['Pearson Correlation'] is not None else '',
                'Kendall\'s Tau': corr['Kendall\'s Tau'] if corr['Kendall\'s Tau'] is not None else '',
                'Spearman': corr['Spearman Correlation'] if corr['Spearman Correlation'] is not None else ''
            })
    
    print(f"Correlation analysis results saved to {output_file}")

if __name__ == "__main__":
    main()