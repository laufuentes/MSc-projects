import numpy as np 
import pandas as pd 


def winetype(data): 
    """Fonction qui repère les indices d'un dataset où la variable "wine_type" est 0 et 1. 

    Args:
        data (pd.DataFrame): Jeu de données à l'origine de la division en fonction de la variable "wine_type"

    Returns:
        index_0 (np.array): vecteur contenant les indices des lignes telles que "wine_type"=0
        index_1 (np.array): vecteur contenant les indices des lignes telles que "wine_type"=1
    """
    wine_type = data[["wine_type"]].to_numpy()
    index_0 = np.where(wine_type==0)[0]
    index_1 = np.where(wine_type==1)[0]
    return index_0,index_1

def treatment(data): 
    """Fonction qui sépare un dataset donné en co-variables (X) et variable explicative (y). 
    NB: cette fonction serà implementée que sur les jeu de données train (data0, data1) 

    Args:
        data (pd.DataFrame): Jeu de données à séparer X, y  

    Returns:
        X (pd.DataFrame): Co-variables du jeu de données
        y (pd.DataFrame): Variable explicative 
    """
    X = data[data.columns[0:-1]]
    y = data[["target"]]
    return X,y


def formal_div(data, idx0, idx1): 
    """Fonction qui sépare le jeu de données data en deux jeux de données data0 (indices où wine_type=0) et data1 (resp. wine_type=1)

    Args:
        data (pd.DataFrame): Jeu de données à séparer en data0 et data1
        idx0 (np.array): Indices où wine_type=0
        idx1 (_type_): Indices où wine_type=1

    Returns:
        data0 (pd.DataFrame): Jeu de données contenant les vins de type 0
        data1 (pd.DataFrame): Jeu de données contenant les vins de type 1
    """
    data0 = data.iloc[idx0,:]
    del data0["wine_type"]
    data1 = data.iloc[idx1,:]
    del data1["wine_type"]
    return data0, data1