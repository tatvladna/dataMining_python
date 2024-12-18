import pandas as pd
from pandas import DataFrame
from rdkit.Chem import Descriptors, SDMolSupplier
from sklearn.preprocessing import  FunctionTransformer


# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================

def load_molecules(file_path):
    return [mol for mol in SDMolSupplier(file_path) if mol is not None]

# ===================================   ПОДГОТОВКА БАЗОВЫХ ДЕСКРИПТОРОВ  ===============================================

# создаем словарь из дескриторов структуры
ConstDescriptors = {"heavy_atom_count": Descriptors.HeavyAtomCount,
                    "nhoh_count": Descriptors.NHOHCount,
                    "no_count": Descriptors.NOCount,
                    "num_h_acceptors": Descriptors.NumHAcceptors,
                    "num_h_donors": Descriptors.NumHDonors,
                    "num_heteroatoms": Descriptors.NumHeteroatoms,
                    "num_rotatable_bonds": Descriptors.NumRotatableBonds,
                    "num_valence_electrons": Descriptors.NumValenceElectrons,
                    "num_aromatic_rings": Descriptors.NumAromaticRings,
                    "num_Aliphatic_heterocycles": Descriptors.NumAliphaticHeterocycles,
                    "ring_count": Descriptors.RingCount}

# создаем словарь из физико-химических дескрипторов                            
PhisChemDescriptors = {"full_molecular_weight": Descriptors.MolWt,
                       "log_p": Descriptors.MolLogP,
                       "molecular_refractivity": Descriptors.MolMR,
                       "tspa": Descriptors.TPSA, # топологическая полярная поверхность
                        "balaban_j": Descriptors.BalabanJ,
                       }

# объединяем все дескрипторы в один словарь
descriptors = {}
descriptors.update(ConstDescriptors)
descriptors.update(PhisChemDescriptors)



# функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols): 
    return DataFrame({k: f(m) for k, f in descriptors.items()} 
                     for m in mols)

descriptors_names = descriptors.keys()

# оформляем sklearn трансформер для использования в конвеерном моделировании (sklearn Pipeline)
descriptors_transformer = FunctionTransformer(mol_dsc_calc, 
                                              validate=False)

def get_descriptors(mols):
    return descriptors_transformer.transform(mols), descriptors_transformer


# Удаляем первые дубликаты и сбрасываем индексы
def remove_duplicates(descriptors):
    descriptors.drop_duplicates(inplace=True)
    descriptors.reset_index(drop=True, inplace=True)
    return descriptors

# ===============================================  ПОДГОТОВКА ВСЕХ ДЕСКРИПТОРОВ =====================================================

# все дескрипторы
def calc_all_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

def get_all_descriptors(calc, mols):
    descriptors_list = [calc(mol) for mol in mols]
    return pd.DataFrame(descriptors_list), FunctionTransformer(calc,validate=False)

def remove_nan(descriptors):
    return descriptors.dropna().reset_index(drop=True)

def remove_data_target(target, list_indices):
    return [value for i, value in enumerate(target) if i not in list_indices]

