import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from sklearn.metrics import (
    root_mean_squared_error, r2_score, f1_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek



log_filename = 'output_data/logs/log_file.log'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename), 
    ]
)

logger = logging.getLogger()

# ============================================  Settings ======================================================
# Настройки pandas`a
def start():
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 35,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+


# ============================================  StandardScaler ================================================

def standard_scaler(transformer, mols):
    standard_scaler = StandardScaler()
    # scaler.fit(base_descriptors.values)
    # base_descriptors_norm = DataFrame(scaler.transform(base_descriptors.values), 
    #                    index=base_descriptors.index, columns=base_descriptors.columns)
    standard_scaler_descriptors_transformer = Pipeline(
        [('descriptors_generation', transformer), ('normalization', standard_scaler)])

    return DataFrame(standard_scaler_descriptors_transformer.fit_transform(mols)), standard_scaler_descriptors_transformer 

# ============================================  MinMaxScaler ======================================================

def min_max_scaler(transformer, mols):
    min_max_scaler = MinMaxScaler()
    # scaler.fit(base_descriptors.values)
    # base_descriptors_norm = DataFrame(scaler.transform(base_descriptors.values), 
    #                    index=base_descriptors.index, columns=base_descriptors.columns)
    min_max_scaler_descriptors_transformer = Pipeline(
        [('descriptors_generation', transformer), ('normalization', min_max_scaler)])

    return DataFrame(min_max_scaler_descriptors_transformer.fit_transform(mols)), min_max_scaler_descriptors_transformer


# =============================================  Lasso ===========================================================

def grid_cv_lasso(x, y, alpha_range=None, cv=5, title_scaler="StandardScaler"):

    logger.info(f"========================== LASSO{title_scaler} для отбора признаков  =========================")
    if title_scaler=="StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    x_scaler = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
        

    if alpha_range is None:
        alpha_range = np.arange(0.0001, 3, 0.01)
    
    lasso = Lasso()
    param_grid = {'alpha': alpha_range}
    
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(x_scaler, y)
    
    optimal_alpha = grid_search.best_params_['alpha']
    logger.info(f"Оптимальный alpha: {optimal_alpha}")
    
    best_lasso = grid_search.best_estimator_
    coefficients = best_lasso.coef_
    logger.info(f"Коэффициенты модели: {coefficients}")
    
    # Отобранные признаки (где коэффициенты = веса не равны нулю)
    numbers_selected_features = [i for i, coef in enumerate(coefficients) if coef != 0]
    selected_features = x_scaler.iloc[:, numbers_selected_features] 
    logger.info(f"Отобранные признаки: {selected_features}")
    

    plt.figure(figsize=(8, 6))
    plt.plot(grid_search.cv_results_['param_alpha'], np.sqrt(-grid_search.cv_results_['mean_test_score']), label='Root Mean test MSE', linewidth=2)
    plt.axvline(optimal_alpha, linestyle='--', color='k', label='Optimal alpha')
    plt.xscale('log') # масштабируем
    plt.xlabel('Alpha')
    plt.ylabel('Root Mean squared error')
    plt.title(f'GridSearchCV: Выбор гиперпараметра alpha для Lasso{title_scaler}')
    plt.legend()
    # plt.show() # сохраняем в папку

    os.makedirs("output_data/lasso", exist_ok=True)
    save_path = os.path.join("output_data/lasso", f"Lasso{title_scaler}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return optimal_alpha, best_lasso, coefficients, selected_features, scaler


# =========================================== GridSearchCV =========================================================

def grid_cv(model, title, param_grid, descriptors, target, size, state, task='regression', sampling = "NonSampl", transformer=None):

    """
    GridSearchCV: 
        1. Возвращает метрики, графики обученных моделей
        2. Сохраняет наилучшую модель и трансформер в файлы формата .pkl.
    
    :param model: Модель для обучения.
    :param title: Название модели.
    :param param_grid: Сетка гиперпараметров для GridSearchCV.
    :param descriptors: Признаки (X).
    :param target: Целевая переменная (y).
    :param size: Размер тестовой выборки.
    :param state: Random state для воспроизводимости.
    :param task: Задача ('regression' или 'classification').
    :param sampling: Метод семплирования ('SMOTE', 'UnderSampl', 'SMOTETomek', 'NonSampl').
    :param transformer: Предобученный трансформер для масштабирования данных.
    """


    x_train, x_test, y_train, y_test = train_test_split(descriptors, target, test_size=size, random_state=state)

    if task == 'regression':
        title = "Ridge" + title + f"_{size:.0%}"
        logger.info(f"========================== Model: Ridge{title}   ===============================")
        scoring = 'r2'
    if task == 'classification':
        title = "LogR" + title + f"_{size:.0%}"
        logger.info(f"========================== Model: LogiscticRegression{title}   ===============================")
        scoring = 'f1'
        if sampling == "SMOTE":
            logger.info(f"Размеры до SMOTE: {x_train.shape}, {len(y_train)}")
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            logger.info(f"Размеры после SMOTE: {x_train.shape}, {len(y_train)}")
        elif sampling == "UnderSampl":
            logger.info(f"Размеры до UnderSampling: {x_train.shape}, {len(y_train)}")
            rus = RandomUnderSampler(random_state=42)
            x_train, y_train = rus.fit_resample(x_train, y_train)
            logger.info(f"Размеры после UnderSampling: {x_train.shape}, {len(y_train)}")
        elif sampling == "SMOTETomek":
            logger.info(f"Размеры до SMOTETomek: {x_train.shape}, {len(y_train)}")
            smote_tomek = SMOTETomek(random_state=42)
            x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
            logger.info(f"Размеры после SMOTETomek: {x_train.shape}, {len(y_train)}")

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        cv=5
    )
    try:
        grid.fit(x_train, y_train)
    except Exception as e:
        logger.exception("Ошибка при обучении модели: %s", e)
        return -1, -1, -1, -1
    
    # ======  ЛУЧШАЯ МОДЕЛЬ ========
    best_model = grid.best_estimator_

    pipeline = Pipeline([
    ('scaler', transformer),  # Добавляем трансформер
    ('model', best_model)  # Добавляем модель
    ])

    os.makedirs("output_data/models", exist_ok=True)
    pipeline_filename = f"output_data/models/pipeline_{title}.pkl"

    with open(pipeline_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    logger.info(f"Лучшая модель сохранена в: {pipeline_filename}")


    logger.info(f'|best params|: {grid.best_params_}')
    logger.info(f'|best fit time|: {grid.refit_time_}')

    y_pred = grid.predict(x_test)

    if task == 'regression':
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        logger.info(f'R2 on test set: {r2}')
        logger.info(f'RMSE on test set: {rmse}')

        # иначе не заработает
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        # эталонная линия
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}: True vs Predicted')
        # plt.show() # не выводим, а сохраняем в папку

        # создаем папку, если она не существует
        os.makedirs('output_data/ridge', exist_ok=True)
        plt.savefig(f'output_data/ridge/{title}.png', dpi=300, bbox_inches='tight')

        return round(r2, 2), round(rmse, 2), grid.refit_time_, grid.cv_results_['mean_score_time'][grid.best_index_], grid.best_params_
    elif task == 'classification':

        f1 = f1_score(y_test, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, grid.predict_proba(x_test)[:, 1])

        logger.info(f'F1 Score: {f1}')
        logger.info(f'Balanced Accuracy: {balanced_acc}')
        logger.info(f'AUC: {auc}')

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test)).plot(cmap='Blues')
        plt.title('Confusion Matrix')
        
        os.makedirs('output_data/logistic_regression/matrix', exist_ok=True)
        plt.savefig(f'output_data/logistic_regression/matrix/{title}.png', dpi=300, bbox_inches='tight')

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(x_test)[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve {title} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        # plt.show()
        os.makedirs('output_data/logistic_regression/roc', exist_ok=True)
        plt.savefig(f'output_data/logistic_regression/roc/{title}.png', dpi=300, bbox_inches='tight')

        return f1, balanced_acc, auc, grid.refit_time_, grid.cv_results_['mean_score_time'][grid.best_index_], grid.best_params_