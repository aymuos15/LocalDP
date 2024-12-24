from runner import train, test
from model import Net_28
from load import load_and_prepare_data, load_and_prepare_noisy_data
from load import load_and_prepare_data_one_pixel
from config import CONFIG

import warnings
import pandas as pd

import tqdm

warnings.filterwarnings("ignore")

def train_and_evaluate(loader_name):
    loaders, class_counts = load_and_prepare_data()
    loaders_noise, _ = load_and_prepare_data_one_pixel()
    # loaders_noise, _ = load_and_prepare_noisy_data()

    model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    model = train(model, loaders[loader_name], CONFIG['task'])   
    acc, auc, class_acc = test(model, loaders['test'], CONFIG['task'])
    del model

    model_noise = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    model_noise = train(model_noise, loaders_noise[loader_name], CONFIG['task'])
    acc_noise, auc_noise, class_acc_noise = test(model_noise, loaders_noise['test'], CONFIG['task'])
    
    return acc, auc, class_acc, acc_noise, auc_noise, class_acc_noise, class_counts[loader_name], class_counts['test']

def main():
    dataset_name = CONFIG['data_path'].split('/')[-1].split('.')[0]
    class_results = []
    class_results_noise = []
    overall_results = []
    overall_results_noise = []
    sample_info = []

    # for loader_name in ['Full', 'LargeSplit', 'SmallSplit']:
    # for loader_name in tqdm.tqdm(['Full', 'LargeSplit', 'SmallSplit']):
    for loader_name in tqdm.tqdm(['Full']):
        acc, auc, class_acc, acc_noise, auc_noise, class_acc_noise, train_class_counts, test_class_counts = train_and_evaluate(loader_name)
        
        for label in range(CONFIG['num_classes']):
            class_results.append({
                'Dataset': dataset_name,
                # 'Loader': loader_name,
                'Class': label,
                'Accuracy': class_acc.get(label, 0),
            })
        
        for label in range(CONFIG['num_classes']):
            class_results_noise.append({
                'Dataset': dataset_name,
                # 'Loader': loader_name,
                'Class': label,
                'Accuracy LocalDP': class_acc_noise.get(label, 0),
            })

        overall_results.append({
            'Dataset': dataset_name,
            # 'Loader': loader_name,
            'Overall Accuracy': acc,
            'Overall AUC': auc,
        })

        overall_results_noise.append({
            'Dataset': dataset_name,
            # 'Loader': loader_name,
            'Overall Accuracy LocalDP': acc_noise,
            'Overall AUC LocalDP': auc_noise,
        })

        sample_info.append({
            # 'Loader': loader_name,
            'Total Train Samples': int(sum(train_class_counts)),
            'Total Test Samples': int(sum(test_class_counts)),
            **{f'Train Samples Class {i}': int(train_class_counts[i]) for i in range(CONFIG['num_classes'])},
            **{f'Test Samples Class {i}': int(test_class_counts[i]) for i in range(CONFIG['num_classes'])}
        })

    class_results_df = pd.DataFrame(class_results)
    overall_results_df = pd.DataFrame(overall_results)
    class_results_noise_df = pd.DataFrame(class_results_noise)
    overall_results_noise_df = pd.DataFrame(overall_results_noise)
    sample_info_df = pd.DataFrame(sample_info).T  # Transpose the sample info DataFrame

    # Combine the Class-wise Results DataFrames
    class_results_df = class_results_df.merge(class_results_noise_df, on=['Dataset', 'Class'], how='outer')

    # Combine the Overall Results DataFrames
    overall_results_df = overall_results_df.merge(overall_results_noise_df, on=['Dataset'], how='outer')
    # Rearrange the columns: 1 2 3 4 5 -> 1 2 4 3 5
    overall_results_df = overall_results_df[['Dataset', 'Overall Accuracy', 'Overall Accuracy LocalDP', 'Overall AUC', 'Overall AUC LocalDP']]
    
    # Display the Class-wise Results DataFrame
    print("\n")
    print("Class-wise Results Table:")
    print(class_results_df.to_string(index=False))
    print("\n")

    # Display the Overall Results DataFrame
    print("Overall Results Table:")
    print(overall_results_df.to_string(index=False))
    print("\n")

    print(sample_info_df.to_string(index=True))

    # Optionally, you can save the DataFrames to CSV files
    # class_results_df.to_csv(CONFIG['result_path'] + 'class_results.csv', index=False)
    # overall_results_df.to_csv(CONFIG['result_path'] + 'overall_results.csv', index=False)
    # class_results_noise_df.to_csv(CONFIG['result_path'] + 'class_results_noise.csv', index=False)
    # overall_results_noise_df.to_csv(CONFIG['result_path'] + 'overall_results_noise.csv', index=False)
    # sample_info_df.to_csv(CONFIG['result_path'] + 'sample_info.csv')

if __name__ == '__main__':
    main()