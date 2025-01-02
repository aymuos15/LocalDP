import pandas as pd
import sys

sys.path.append('/home/localssk23/localdp/')
from config import CONFIG

# Get Data
result_path = CONFIG['result_path']
dataset_name = CONFIG['data_path'].split('/')[-1].split('.')[0]

print(f"Results for {dataset_name}")
print()

#################
# Overall Score #
#################

overall_table = pd.read_csv(result_path + dataset_name + '_' + 'overall_results.csv')

# Correct column names for AUC and Accuracy
auc_columns = ['Overall_AUC_original', 'Overall_AUC_noisy_1pixel'] + [f'Overall_AUC_noisy_{i}%' for i in [1, 5, 10, 25, 50, 75, 90, 99]]
acc_columns = ['Overall_Accuracy_original', 'Overall_Accuracy_noisy_1pixel'] + [f'Overall_Accuracy_noisy_{i}%' for i in [1, 5, 10, 25, 50, 75, 90, 99]]

# Create separate tables
auc_table = pd.DataFrame({
    'Noise Level': ['Original', '1pixel', '1', '5', '10', '25', '50', '75', '90', '99'],
    'AUC Score': [overall_table[col].values[0] for col in auc_columns]
})

acc_table = pd.DataFrame({
    'Noise Level': ['Original', '1pixel', '1', '5', '10', '25', '50', '75', '90', '99'],
    'Accuracy': [overall_table[col].values[0] for col in acc_columns]
})

# combine tables
combined_table = pd.concat([auc_table, acc_table['Accuracy']], axis=1)

print(combined_table.to_string(index=False))
print()

####################
# Classwise Scores #
####################

class_table = pd.read_csv(result_path + dataset_name + '_' + 'class_results.csv')

# Create the DataFrame from the original data
df = pd.DataFrame(class_table)

# Create DataFrame and transpose
df = pd.DataFrame(class_table).drop(columns=['Split'])
transposed_df = df.set_index(["Class"]).transpose()

# Reset index and rename column
transposed_df.reset_index(inplace=True)
transposed_df.rename(columns={"index": "Metric"}, inplace=True)

# Drop the first row   
transposed_df = transposed_df[1:]

print(transposed_df)