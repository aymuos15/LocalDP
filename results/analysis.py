import pandas as pd
import numpy as np
from pathlib import Path

import sys

sys.path.append('/home/localssk23/localdp/')
from config import CONFIG

def format_overall_results(dataset_name, result_path, num_folds=3):
    all_folds_auc = []
    all_folds_acc = []

    for fold in range(num_folds):
        overall_file = f"{dataset_name}_overall_results_fold_{fold}.csv"
        overall_table = pd.read_csv(Path(result_path) / overall_file)
        
        # Updated column names to match your CSV
        auc_columns = ['Overall_AUC_None', 'Overall_AUC_one'] + \
                     [f'Overall_AUC_{i}' for i in [1, 5, 10, 25, 50, 75, 90, 99]]
        acc_columns = ['Overall_Accuracy_None', 'Overall_Accuracy_one'] + \
                     [f'Overall_Accuracy_{i}' for i in [1, 5, 10, 25, 50, 75, 90, 99]]
        
        fold_auc = pd.DataFrame({
            'Noise Level': ['Original', '1pixel', '1', '5', '10', '25', '50', '75', '90', '99'],
            f'AUC Score Fold {fold}': [overall_table[col].values[0] for col in auc_columns]
        })
        
        fold_acc = pd.DataFrame({
            'Noise Level': ['Original', '1pixel', '1', '5', '10', '25', '50', '75', '90', '99'],
            f'Accuracy Fold {fold}': [overall_table[col].values[0] for col in acc_columns]
        })
        
        all_folds_auc.append(fold_auc)
        all_folds_acc.append(fold_acc)

    combined_auc = all_folds_auc[0]
    combined_acc = all_folds_acc[0]
    for i in range(1, num_folds):
        combined_auc = pd.merge(combined_auc, all_folds_auc[i], on='Noise Level')
        combined_acc = pd.merge(combined_acc, all_folds_acc[i], on='Noise Level')
    
    auc_cols = [col for col in combined_auc.columns if 'AUC' in col]
    acc_cols = [col for col in combined_acc.columns if 'Accuracy' in col]
    
    combined_auc['Mean AUC'] = combined_auc[auc_cols].mean(axis=1)
    combined_acc['Mean Accuracy'] = combined_acc[acc_cols].mean(axis=1)
    
    return combined_auc, combined_acc

def format_class_results(dataset_name, result_path, num_folds=3):
    noise_levels = ['Original', '1pixel'] + [f'{i}' for i in [1, 5, 10, 25, 50, 75, 90, 99]]
    
    fold_data = []
    for fold in range(num_folds):
        class_file = f"{dataset_name}_class_results_fold_{fold}.csv"
        df = pd.read_csv(f"{result_path}/{class_file}")
        fold_data.append(df)
    
    num_classes = len(fold_data[0]['Class'].unique())
    results = {level: {f'class{i}': [] for i in range(num_classes)} for level in noise_levels}
    
    for fold_idx, df in enumerate(fold_data):
        for noise in noise_levels:
            if noise == 'Original':
                col_name = 'Accuracy_None'
            elif noise == '1pixel':
                col_name = 'Accuracy_one'
            else:
                col_name = f'Accuracy_{noise}'
            
            for class_idx in range(num_classes):
                class_value = df[df['Class'] == class_idx][col_name].values[0]
                results[noise][f'class{class_idx}'].append(class_value)
    
    data = []
    for noise in noise_levels:
        row = [noise]
        for fold in range(num_folds):
            for class_idx in range(num_classes):
                row.append(results[noise][f'class{class_idx}'][fold])
        for class_idx in range(num_classes):
            row.append(np.mean(results[noise][f'class{class_idx}']))
        data.append(row)
    
    columns = ['Noise Level']
    for fold in range(num_folds):
        columns.extend([f'Class {i} Fold {fold}' for i in range(num_classes)])
    columns.extend([f'Mean Class {i}' for i in range(num_classes)])
    
    return pd.DataFrame(data, columns=columns)

def create_summary_table(class_results):
   # Get noise levels from first column
   noise_levels = class_results['Noise Level']
   
   # Get number of classes by looking at mean columns
   num_classes = sum(1 for col in class_results.columns if 'Mean Class' in col)
   
   # Create summary data
   summary_data = []
   for idx, noise in enumerate(noise_levels):
       row = [noise]
       for class_idx in range(num_classes):
           # Get columns for this class across all folds
           fold_cols = [col for col in class_results.columns if f'Class {class_idx} Fold' in col]
           mean_value = class_results.loc[idx, fold_cols].mean()
           row.append(mean_value)
       summary_data.append(row)
       
   columns = ['Noise Level'] + [f'Class {i} Mean' for i in range(num_classes)]
   return pd.DataFrame(summary_data, columns=columns)

def main():
   # Configure paths and datasets
   result_path = CONFIG['result_path']
   # dataset_names = ['breastmnist', 'retinamnist', 'bloodmnist']  # Add more datasets as needed
   dataset_names = ['breastmnist']  # Add more datasets as needed

   num_folds = 2

   for dataset_name in dataset_names:
       print(f"\nResults for {dataset_name}")
       print("=" * 50)
       
       # Get overall results
       combined_auc, combined_acc = format_overall_results(dataset_name, result_path, num_folds)
       print("\nOverall AUC Scores:")
       print(combined_auc.to_string(index=False))
       print("\nOverall Accuracy Scores:")
       print(combined_acc.to_string(index=False))
       
       # Get class-wise results
       class_results = format_class_results(dataset_name, result_path, num_folds)
       print("\nClass-wise Accuracy Scores:")
       print(class_results.to_string(index=False))

       # Fold wise class wise mean    
       summary_results = create_summary_table(class_results)
       print("\nSummary Across All Folds:")
       print(summary_results.to_string(index=False))
       print()

if __name__ == "__main__":
   main()