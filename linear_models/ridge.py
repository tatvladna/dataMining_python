from rdkit.Chem import Descriptors, SDMolSupplier
from sklearn.linear_model import LogisticRegression, Ridge
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


# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================

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

# =========================================  ИЗВЛЕЧЕНИЕ ТАРГЕТА =============================================
target_logbcf = [m.GetDoubleProp('logBCF') for m in molecules]

# удаляем объекты, где есть nan
target_logbcf_lasso = remove_data_target(target_logbcf, list_indices_nan)

# ============================================ ОТБОР ДЕСКРИПТОРОВ С ПОМОЩЬЮ LASSO ===================================================

optimal_alpha_standard_scaler, _ , coefficients, features_lasso_standard_scaler, standard_scaler_lasso = grid_cv_lasso(all_descriptors, 
                                                                                                target_logbcf_lasso, 
                                                                                                title_scaler="StandardScaler")

optimal_alpha_min_max_scaler, _ , coefficients, features_lasso_min_max_scaler, min_max_scaler_lasso = grid_cv_lasso(all_descriptors, 
                                                                                              target_logbcf_lasso, 
                                                                                              title_scaler="MinMaxScaler")

# ====================================== ПОСТРОЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ ==========================================================

ridge_model = Ridge(max_iter=10000) # используется в 'lsqr', 'saga', игнорируется в auto, svd, lsqr, и cholesky

param_ridge = {
    'alpha': np.arange(0.01, 80, 0.01),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'] 
}


results_ridge = []

sizes = [1/3, 1/4, 1/5, 1/10]
list_descriptors = {"MinMaxSc": base_descriptors_min_max_scaler, "StdSc": base_descriptors_standard_scaler, 
                    "LassoStdSc": features_lasso_standard_scaler, "LassoMinMaxSc": features_lasso_min_max_scaler}

transformer = ""

for title, descriptors in list_descriptors.items():
    target_logbcf = [m.GetDoubleProp('logBCF') for m in molecules]
    
    param_ridge = {
    'alpha': np.arange(0.01, 20, 0.01),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr',"saga"]
}

    if title == "LassoStdSc":
        # для Lasso удаляли объекты
        target_logbcf = target_logbcf_lasso
        param_ridge = {
            'alpha': [optimal_alpha_standard_scaler],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
        transformer = standard_scaler_lasso
    elif title == "LassoMinMaxSc":
            # для Lasso удаляли объекты
            target_logbcf = target_logbcf_lasso
            param_ridge = {
            'alpha': [optimal_alpha_min_max_scaler],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        }
            transformer = min_max_scaler_lasso
    elif title=="MinMaxSc":
         transformer = base_descriptors_min_max_scaler_transformer
    elif title=="StdSc":
         transformer = base_descriptors_standard_scaler_transformer


    for size in sizes:                    
        log_r2, log_rmse, fit_time, predict_time, best_params  = grid_cv(model = ridge_model,
                                                                            title = title, 
                                                                            param_grid = param_ridge,
                                                                            descriptors= descriptors, 
                                                                            target= target_logbcf, 
                                                                            size=size, 
                                                                            state=RANDOM_STATE,
                                                                            task = 'regression',
                                                                            transformer=transformer)

        results_ridge.append({
            "Model": f"Ridge{title}",
            "Размер тестовой выборки": f"{int(size*100)}%",
            "R2": log_r2,
            "RMSE": log_rmse,
            "Время обучения (сек)": fit_time,
            "Время предсказания (сек)": predict_time,
            "Качество модели": log_r2 >= 0.65 and log_rmse <= 0.7,
            "Гиперпараметры": best_params
        })

table_scan = pd.DataFrame(results_ridge)


output_folder_ridge = "results_ridge"
os.makedirs(output_folder_ridge, exist_ok=True)
txt_path_ridge = os.path.join(output_folder_ridge, 'ridge_logbcf.txt')
csv_path_ridge = os.path.join(output_folder_ridge, 'ridge_logbcf.csv')

with open(txt_path_ridge, 'w', encoding='utf-8') as file:
    file.write(table_scan.to_string(index=False))

table_scan.to_csv(csv_path_ridge, index=False, encoding='utf-8')
