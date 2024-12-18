import argparse
import os
import pickle
parser = argparse.ArgumentParser(description='Выбор модели и обработка данных для предсказаний.')
parser.add_argument('--model', '-m', type=str, choices=['ridge', 'logistic', 'tree'], required=True, 
                    help='Выбор модели: ridge, logistic, tree')
parser.add_argument('--input', '-i', type=str, required=True, 
                    help='Входной sdf-файл с молекулой для предсказания logBCF')
args = parser.parse_args()


if os.path.isfile(args.input):
    print("file found")
else:
    print(f"Error: file {args.input} not found")
    exit(1) # завершаем программу


# Пример
with open("linear_model/output_data/models/best_model_RidgeMinMaxSc.pkl", 'rb') as model_file:
    best_model = pickle.load(model_file)

# Пример
with open("linear_model/output_data/models/transformer_RidgeMinMaxSc.pkl", 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)


