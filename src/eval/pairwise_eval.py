import os
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(input_folder_path, output_csv_path="metrics_results.csv"):
    """
    Calculate evaluation metrics for each model subfolder: Precision, Recall, F1, Consistency, Acc (w/swap.)
    
    Args:
        input_folder_path (str): Root directory path containing model subfolders and JSON files
        output_csv_path (str): Complete path for the output CSV file
    """
    # Store metric results for each model
    results = {}
    
    # Traverse all model subfolders under the root directory
    for model_name in os.listdir(input_folder_path):
        model_dir_path = os.path.join(input_folder_path, model_name)
        
        # Skip non-directory items
        if not os.path.isdir(model_dir_path):
            continue
        
        # Build complete paths for forward.json and reverse.json
        forward_path = os.path.join(model_dir_path, 'forward.json')
        reverse_path = os.path.join(model_dir_path, 'reverse.json')
        
        # Check if files exist
        if not (os.path.exists(forward_path) and os.path.exists(reverse_path)):
            print(f"Warning: Complete forward.json and reverse.json files not found in {model_name}, skipping.")
            continue
        
        # Read JSON files
        try:
            # Load forward.json (original order evaluation)
            forward_data = []
            with open(forward_path, 'r', encoding='utf-8') as f:
                # Check if it's a JSONL file
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                if first_line.startswith('{') and first_line.endswith('}'):
                    # Read JSONL format line by line
                    for line in f:
                        if line.strip():
                            forward_data.append(json.loads(line))
                else:
                    # Read the entire file as a single JSON
                    forward_data = json.load(f)
            
            # Load reverse.json (swapped order evaluation)
            reverse_data = []
            with open(reverse_path, 'r', encoding='utf-8') as f:
                # Check if it's a JSONL file
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                if first_line.startswith('{') and first_line.endswith('}'):
                    # Read JSONL format line by line
                    for line in f:
                        if line.strip():
                            reverse_data.append(json.loads(line))
                else:
                    # Read the entire file as a single JSON
                    reverse_data = json.load(f)
            
            # Ensure data counts match
            if len(forward_data) != len(reverse_data):
                print(f"Warning: Data counts in forward.json and reverse.json do not match in {model_name}, skipping.")
                continue
            
            # Prepare data for new calculation logic
            true_labels = []
            predicted_labels = []
            consistency_count = 0  # Consistency count
            accuracy_with_swap_count = 0  # Accuracy after swap count
            total_samples = len(forward_data)
            unknown_count = 0  # Count unknown predictions
            
            # Process each data pair
            for i in range(len(forward_data)):
                # Get required fields (compatible with multiple field names)
                forward_pred = forward_data[i].get('pred', forward_data[i].get('result', 'unknown'))
                forward_label = forward_data[i].get('label')
                reverse_pred = reverse_data[i].get('pred', reverse_data[i].get('result', 'unknown'))
                
                # Skip any data with missing labels
                if forward_label is None:
                    continue
                
                # Count unknown predictions
                if forward_pred == "unknown" or forward_pred == -1:
                    unknown_count += 1
                    forward_pred = "unknown"
                if reverse_pred == "unknown" or reverse_pred == -1:
                    unknown_count += 1
                    reverse_pred = "unknown"
                
                # Prepare data for Precision, Recall, F1
                # Handle unknown cases
                if forward_pred == "unknown":
                    # Replace unknown with a value that is definitely different from the true label
                    if forward_label == "A":
                        processed_pred = "B"
                    elif forward_label == "B":
                        processed_pred = "A"
                    else:  # forward_label is Tie
                        processed_pred = "A"
                else:
                    processed_pred = forward_pred
                
                true_labels.append(forward_label)
                predicted_labels.append(processed_pred)
                
                # Calculate Consistency
                is_consistent = False
                if forward_pred == "unknown" or reverse_pred == "unknown":
                    # unknown is always considered inconsistent
                    pass
                elif (forward_pred == "A" and reverse_pred == "B") or \
                     (forward_pred == "B" and reverse_pred == "A") or \
                     (forward_pred == "Tie" and reverse_pred == "Tie"):
                    is_consistent = True
                    consistency_count += 1
                
                # Calculate Acc (w/swap.)
                is_accurate_with_swap = False
                if forward_pred == "unknown" or reverse_pred == "unknown":
                    # unknown is always considered inaccurate
                    pass
                elif forward_label == "Tie":
                    if forward_pred == "Tie" and reverse_pred == "Tie":
                        is_accurate_with_swap = True
                else:  # forward_label is A or B
                    # forward prediction needs to match label, reverse prediction needs to be opposite of label
                    opposite_label = "B" if forward_label == "A" else "A"
                    if forward_pred == forward_label and reverse_pred == opposite_label:
                        is_accurate_with_swap = True
                
                if is_accurate_with_swap:
                    accuracy_with_swap_count += 1
            
            # Use sklearn to calculate Precision, Recall, and F1
            # Ensure labels are unique for sklearn processing
            unique_labels = list(set(true_labels + predicted_labels))
            precision = precision_score(true_labels, predicted_labels, 
                                        labels=unique_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predicted_labels, 
                                 labels=unique_labels, average='macro', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, 
                         labels=unique_labels, average='weighted', zero_division=0)
            
            # Calculate Consistency and Acc (w/swap.)
            consistency = consistency_count / total_samples
            acc_with_swap = accuracy_with_swap_count / total_samples
            
            # Calculate Agreement (simple consistency)
            agreement_count = 0
            for i in range(len(forward_data)):
                forward_pred = forward_data[i].get('pred', forward_data[i].get('result', 'unknown'))
                if forward_pred == -1:
                    forward_pred = "unknown"
                    
                forward_label = forward_data[i].get('label')
                
                if forward_pred != "unknown" and forward_pred == forward_label:
                    agreement_count += 1
            
            agreement = agreement_count / total_samples
            
            # Store model results
            results[model_name] = {
                'Agreement': agreement,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Consistency': consistency,
                'Acc (w/swap.)': acc_with_swap,
                'Unknown Count': unknown_count
            }
            
            print(f"Calculated metrics for {model_name}:")
            print(f"  Total samples: {total_samples}")
            print(f"  Unknown predictions: {unknown_count}")
            print(f"  Agreement: {agreement:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Consistency: {consistency:.4f}")
            print(f"  Acc (w/swap.): {acc_with_swap:.4f}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create results table
    if results:
        # Create DataFrame for more flexible data handling
        df_results = pd.DataFrame()
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Agreement': metrics['Agreement'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'Consistency': metrics['Consistency'],
                'Acc (w/swap.)': metrics['Acc (w/swap.)'],
                'Unknown Count': metrics['Unknown Count']
            }
            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
        
        # Calculate average values
        avg_row = {
            'Model': 'Average',
            'Agreement': df_results['Agreement'].mean(),
            'Precision': df_results['Precision'].mean(),
            'Recall': df_results['Recall'].mean(),
            'F1': df_results['F1'].mean(),
            'Consistency': df_results['Consistency'].mean(),
            'Acc (w/swap.)': df_results['Acc (w/swap.)'].mean(),
            'Unknown Count': df_results['Unknown Count'].sum()
        }
        df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Print results table
        print("\nMetrics Summary Table:")
        print("-" * 100)
        
        # Format output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df_results)
        
        print("-" * 100)
        
        # Save results to CSV file
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        df_results.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"\nResults saved to {output_csv_path}")
        
        return df_results
    else:
        print("No valid model subfolders found for calculation.")
        return pd.DataFrame()


def main():
    """Main function, process command line arguments and call calculation function"""
    parser = argparse.ArgumentParser(description='Calculate LLM evaluation metrics and generate summary report')
    
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Root directory path containing model evaluation results')
    
    parser.add_argument('--output_csv', '-o', type=str, default='metrics_summary.csv',
                        help='Path for output CSV file (default: metrics_summary.csv)')
    
    args = parser.parse_args()
    
    # Print startup information
    print(f"Starting evaluation data analysis...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output CSV file: {args.output_csv}")
    print("-" * 50)
    
    # Calculate metrics
    calculate_metrics(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()