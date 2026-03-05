from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np 
import csv


def CV_rep(Xtr, ytr, nfolds): 
    """La fonction crée des nouveaux jeux de données en sous-divisant les jeux de données en 

    Args:
        Xtr (pd.DataFrame): Jeu de données à diviser, contenant les co-variables 
        ytr (pd.DataFrame): Vecteur à diviser, contenant la variable à prédire
        nfolds (int): Nombre representant en combien de sous dataframes on souhaite diviser Xtr et ytr

    Returns:
        X_new (list): liste contenant (n-folds) jeux de données 
        y_new (list): liste contenant (les n-folds) nouvelles version de ytr
    """
    r = np.random.randint(low=0,high=nfolds, size=Xtr.shape[0])
    X_new = []
    y_new = []
    for i in range(nfolds): 
        index = np.where(r==i)[0]
        X_new.append(Xtr.iloc[index])
        y_new.append(ytr.iloc[index])
    return X_new, y_new

def Label_Encode(y_tr, new_pred): 
    """Fonction qui normalise les etiquettes de plusieurs jeux de données

    Args:
        y_tr (pd.DataFrame): vecteur qui contient les etiquettes que l'on utilisera pour entrainer le modèle
        new_pred (np.array): vecteur auquel on appliquera la normalisation des etiquettes

    Returns:
        new_pred (np.array): même vecteur qu'avant avec les etiquettes normalisées
    """
    le = LabelEncoder()
    le.fit(y_tr)
    new_pred = le.transform(new_pred)
    return new_pred

def submission(new_pred,name, date, X_test): 
    """Fonction qui genere un csv avec les valeurs à prédire associées à ses respectifs identificateurs 
    Attention: il faudrait aussi verifier que obj_ID se retrouve dans le repertoire nommé
    
    Args:
        new_pred (np.array): vecteur contenant les predictions
        name (string): méthode utilisée pour generer la prédiction (ex: 'LDA')
        date (string): date de la prédiction (ex: '04/10')
        X_test (pd.DataFrame): Jeu de données test utilisé pour calculer la prédiction 
    """
    data = np.column_stack((X_test["obj_ID"], new_pred))
    # Nommez les colonnes
    column_names = ['obj_ID', 'label']
    # Spécifiez le nom du fichier CSV de sortie
    csv_filename = 'submissions_C/'+name+'_'+date+'.csv'
    # Ouvrez le fichier CSV en mode écriture et écrivez les données
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Écrivez les noms des colonnes
        writer.writerow(column_names)
        # Écrivez les données
        writer.writerows(data)
    return("OK") 