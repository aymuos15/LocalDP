import warnings
from utils import process_results, train_and_evaluate
from config import CONFIG, PERTURBATION_VALUES

warnings.filterwarnings("ignore")

def main(fold: int):
    dataset_name = CONFIG['data_path'].rsplit('/', 1)[-1].rsplit('.', 1)[0]
    print(f'DATA: {dataset_name}')
    
    results = train_and_evaluate(PERTURBATION_VALUES)
    process_results(results, dataset_name, fold)

if __name__ == '__main__':
    for fold in range(CONFIG['num_folds']):
        main(fold)
        print(f'Completed fold {fold}\n')