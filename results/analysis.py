import pandas as pd
import sys
sys.path.append('/home/localssk23/splits_mnist/')
from config import CONFIG

class_results = pd.read_csv(CONFIG['result_path'] + 'class_results.csv')
overall_results = pd.read_csv(CONFIG['result_path'] + 'overall_results.csv')
class_results_corrupted = pd.read_csv(CONFIG['result_path'] + 'class_results_corrupted.csv')
overall_results_corrupted = pd.read_csv(CONFIG['result_path'] + 'overall_results_corrupted.csv')

# Add the last column from class_results_corrupted to class_results
class_results = pd.concat([class_results, class_results_corrupted.iloc[:, -1]], axis=1)
overall_results = pd.concat([overall_results, overall_results_corrupted.iloc[:, -2]], axis=1)
overall_results = pd.concat([overall_results, overall_results_corrupted.iloc[:, -1]], axis=1)

# Swap the position of column 4 and 5 in overall_results
cols = overall_results.columns.tolist()
cols[3], cols[4] = cols[4], cols[3]
overall_results = overall_results[cols]

# Convert dataframes to strings
class_results = class_results.to_string()
overall_results = overall_results.to_string()

print(class_results)
print()
print(overall_results)
