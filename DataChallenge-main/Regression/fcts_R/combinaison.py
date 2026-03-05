import numpy as np 
import pandas as pd 
from fcts_R.general import * 
from fcts_R.dataset_division import *
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import * 

def liste_preds(models, X_tr,y_tr, X_te):
    """Fonction qui calcule les prédicteurs associées a chaque modèle dans models 

    Args:
        models (list): vecteur contenant plusieurs modèles
        X_tr (pd.DataFrame): Jeu de données d'entraînemenent avec les co-variables 
        y_tr (pd.DataFrame ou np.array): Jeu de données d'entraînement avec la variable à prédire 
        X_te (pd.DataFrame): Jeu de données test avec les co-variables sur lesquels on calculera les prédictions 

    Returns:
        liste_ (list): liste contenant l'ensemble des prédicteurs 
    """
    liste_ = []
    for mod in models:
        pipe = Pipeline(steps=[('std', StandardScaler()),('mod', mod)])
        liste_.append(pipe.fit(X_tr,y_tr.to_numpy().ravel()).predict(X_te))
    return liste_

def choix_melange(models_0, models_1, data0, data1): 
    np.random.seed(50)
    """Fonction qui calcule par cross validation la meilleure combinaison de prédicteurs. (pred0 et pred1)
    Sur chaque fold: 
     -  Calcule toutes les combinaisons entre les prédicteurs de data0 et data1
     - Repère le meilleur r2_score issu des combinaisons 
     - Renvoit les indices de cette valeur maximale 

     Enfin, elle renvoit les modèles la combinaison de modèles plus votée 

    Args:
        models_0 (np.array): Vecteur avec l'ensemble des modèles pour data0 à combiner
        models_1 (np.array): Vecteur avec l'ensemble des modèles pour data1 à combiner
        data0 (pd.DataFrame): Jeu de données ayant wine_type=0
        data1 (pd.DataFrame): Jeu de données ayant wine_type=1

    Returns:
        _type_: _description_
    """
    X0,y0 = treatment(data0)
    X1,y1 = treatment(data1)
    folds = 10
    X_0, y_0 = CV_rep(X0, y0, folds)
    X_1, y_1 = CV_rep(X1, y1, folds)
    b_0 = np.zeros(3)

    for k in range(folds): 
        res = np.zeros((len(models_0),len(models_1)))
        X_tr0, X_te0, y_tr0, y_te0 = train_test_split(X_0[k],y_0[k], test_size=0.33, random_state=42)
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_1[k],y_1[k], test_size=0.33, random_state=42)
        preds_0 = liste_preds(models_0,X_tr0, y_tr0, X_te0)
        preds_1 = liste_preds(models_1,X_tr1, y_tr1, X_te1)
        for i, ii in enumerate(preds_0): 
            for j,jj in enumerate(preds_1): 
                pred = build_pred(X_te0, X_te1, ii,jj)[:,1]
                res[i,j] = r2_score(build_pred(X_te0, X_te1, y_te0, y_te1)[:,1], pred)
        maxx = np.unravel_index(np.argmax(res, axis=None), res.shape)
        b_0 = np.vstack ((b_0, np.array([np.max(res), maxx[0], maxx[1]])))

    #On va donner priorité aux plus forts scores de cross-validation
    b_0 = b_0[np.where(b_0[:,0] > np.sort(b_0[:,0])[5])[0]]
    #On va chercher les répétitions
    matr = np.array([''.join(map(str, ligne)) for ligne in b_0[:,1:]])
    _, occurrences = np.unique(matr, return_counts=True)
    #Indices des modèles choisis par vote majoritaire 
    idx0 = b_0[np.argmax(occurrences), 1]
    idx1 = b_0[np.argmax(occurrences),2]
    return models_0[int(idx0)], models_1[int(idx1)]