from rdkit.Chem import Descriptors, SDMolSupplier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import standard_scaler, grid_cv_lasso, grid_cv, min_max_scaler, standard_scaler
from utils.generate_descriptors import (load_molecules, get_descriptors, 
                                    remove_duplicates, get_all_descriptors, 
                                    calc_all_descriptors, remove_nan,
                                    remove_data_target)

RANDOM_STATE=28112024


# =========================================== ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ ЗАДАЧИ КЛАССИФИКАЦИИ  ==========================================

molecules = load_molecules("../data/learning/logBCF.sdf")

# базовые дескрпиторы, которые были отобраны вручную
base_descriptors, descriptors_transformer = get_descriptors(molecules)
base_descriptors = remove_duplicates(base_descriptors)


# все дескрипторы для Lasso
all_descriptors, all_descriptors_transformer = get_all_descriptors(calc_all_descriptors, molecules)

list_indices_nan = list(all_descriptors[all_descriptors.isna().any(axis=1)].index)

all_descriptors = remove_nan(all_descriptors)


#  ======================================== МАСШТАБИРОВАНИЕ =================================================

base_descriptors_min_max_scaler, base_descriptors_min_max_scaler_transformer = min_max_scaler(descriptors_transformer, molecules)

base_descriptors_standard_scaler, base_descriptors_standard_scaler_transformer = standard_scaler(descriptors_transformer, molecules)


# ========= TARGET ===========

target_logbcf_class = [m.GetIntProp('class') for m in molecules]

# используем те же самые дескрипторы, только таргет в виде классов
target_logbcf_lasso_class = remove_data_target(target_logbcf_class, list_indices_nan)

# ========  LASSO ==========
optimal_alpha_standard_scaler_class, _ , coefficients_class, features_lasso_standard_scaler_class, standard_scaler_lasso = grid_cv_lasso(all_descriptors, 
                                                                                                target_logbcf_lasso_class, 
                                                                                                title_scaler="StandardScaler")

optimal_alpha_min_max_scaler_class, _ , coefficients_class, features_lasso_min_max_scaler_class, min_max_scaler_lasso = grid_cv_lasso(all_descriptors, 
                                                                                              target_logbcf_lasso_class, 
                                                                                              title_scaler="MinMaxScaler")

# ==========  Моделирование =========

logistic_model = LogisticRegression()

results_logistic = []

sizes = [1/3, 1/4, 1/5, 1/10]
# базовые дескрипторы остаются неизменными
list_descriptors = {"MinMaxSc": base_descriptors_min_max_scaler, "StdSc": base_descriptors_standard_scaler, 
                    "LassoStdSc": features_lasso_standard_scaler_class, "LassoMinMaxSc": features_lasso_min_max_scaler_class}

list_sampling = ["NonSampl", "SMOTE", "UnderSampl", "SMOTETomek"]
transformer = ""

for title, descriptors in list_descriptors.items():
    
    target_logbcf_class = [m.GetIntProp('class') for m in molecules]

    # elasticnet поддерживает только solver = saga
    # и для него нужно указывать долю вклада l1 и l2 регуляризации l1_ratio
    param_grid_logistic = {
    'penalty': ['l2'],  # Тип регуляризации
    'C': [0.1, 150, 0.01],  # Сила регуляризации (ипользуется только, если penalty не None)
    'solver': ['lbfgs', 'liblinear', 'saga'],  # Алгоритмы оптимизации
    'max_iter': np.arange(50, 1200, 50),  # Количество итераций
    'class_weight': [None, 'balanced'],  # Веса классов
}
    if title == "LassoStdSc":
        transformer = standard_scaler_lasso

        target_logbcf_class = target_logbcf_lasso_class

        param_grid_logistic = {
        'penalty': ['l2'],  # Тип регуляризации
        'C': [1/ 2 * optimal_alpha_standard_scaler_class],  # Сила регуляризации
        'solver': ['lbfgs', 'liblinear', 'saga'],  # Алгоритмы оптимизации
        'max_iter': np.arange(50, 1200, 50),  # Количество итераций
        'class_weight': [None, 'balanced'],  # Веса классов
        }

    elif title == "LassoMinMaxSc":

        transformer = min_max_scaler_lasso
        target_logbcf_class = target_logbcf_lasso_class

        param_grid_logistic = {
        'penalty': ['l2'],  # Тип регуляризации
        'C': [1/ 2 * optimal_alpha_min_max_scaler_class],  # Сила регуляризации
        'solver': ['lbfgs', 'liblinear', 'saga'],  # Алгоритмы оптимизации
        'max_iter': np.arange(50, 1200, 50),  # Количество итераций
        'class_weight': [None, 'balanced'],  # Веса классов
        }
    elif title=="MinMaxSc":
         transformer = base_descriptors_min_max_scaler_transformer
    elif title=="StdSc":
         transformer = base_descriptors_standard_scaler_transformer

    for sampling in list_sampling:
        for size in sizes:                    
            log_f1, log_balanced_acc, log_auc, fit_time, predict_time, best_params  = grid_cv(model = logistic_model,
                                                                                                title = title + sampling,
                                                                                                param_grid = param_grid_logistic,
                                                                                                descriptors= descriptors, 
                                                                                                target= target_logbcf_class, 
                                                                                                size=size, 
                                                                                                state=RANDOM_STATE,
                                                                                                sampling = sampling,
                                                                                                task = "classification")



            results_logistic.append({
                "Model LogisticRegression": f"LogR{title}{sampling}",
                "Размер тестовой выборки": f"{int(size*100)}%",
                "balanced Accuracy": log_balanced_acc,
                "F1": log_f1,
                "AUC": log_auc,
                "Время обучения (сек)": fit_time,
                "Время предсказания (сек)": predict_time,
                "Качество модели": log_f1 >= 0.75 and log_auc >= 0.75,
                "Параметры": best_params
            })

results_table = pd.DataFrame(results_logistic)

output_folder_log = "results_logistic_regr"
os.makedirs(output_folder_log, exist_ok=True)

txt_path_log = os.path.join(output_folder_log, 'logistic_regr_logbcf.txt')
csv_path_log = os.path.join(output_folder_log, 'logistic_regr_logbcf.csv')
with open(txt_path_log, 'w', encoding='utf-8') as file:
    file.write(results_table.to_string(index=False))

results_table.to_csv(csv_path_log, index=False)