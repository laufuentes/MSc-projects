import numpy as np
import matplotlib.pyplot as plt
import pandas 
from fcts_c.general import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def f1(ytr,pred, lab): 
    """Fonction qui calcule le score f1 associé à une prédiction (pour un problème de classification binaire, C=[3,lab] avec lab={0,2})

    Args:
        ytr (pd.DataFrame): vecteur contenant les labels du jeu de données train
        pred (np.array): vecteur contenant des prédictions (2 classes)
        lab (_type_): _description_

    Returns:
        (2*TP)/(2*TP + FP + FN) (float): valeur du score f1 associé à cette prédiction
    """

    #ytr = ytr.to_numpy()
    index_true = np.where(ytr==lab)[0]
    index_false = np.where(ytr!=lab)[0]
    TP = len(np.where(pred[index_true]==lab)[0])/len(ytr)
    FN = len(np.where(pred[index_true]!=lab)[0])/len(ytr)
    FP = len(np.where(pred[index_false]==lab)[0])/len(ytr)
    return (2*TP)/(2*TP + FP + FN)

def frontiere(Xtr,ytr, lab, folds, nb_seuils, n_var): 
    """Cette fonction crée un vecteur contenant, nb_seuils, seuils différents et teste leur performance (f1 score) 
    pour classfifier le label lab à partir de la variable n_var. On utilisera la méthode de cross validation pour tester les performances 
    moyennes de chaque seuil.  

    Args:
        Xtr (pd.DataFrame): Jeu de données train contenant les co-variables
        ytr (pd.DataFrame): Jeu de données train contenant la variable à prédire
        lab (int: 0 ou 2): label pour lequel calculer la valeur frontière 
        folds (int): nombre de folds a créer pour performer la cross-validation
        nb_seuils (int): nombre de seuils à tester entre la valeur min,max de n_var
        n_var (string): nom de la variable sur laquelle on veut calculer le seuil

    Returns:
        results.mean(axis=0) (np.array): vecteur contenant les f1-scores moyennes de chaque seuil par cross-validation
    """
    seuils = np.linspace(Xtr[[n_var]].min()[0], Xtr[[n_var]].max()[0], nb_seuils)
    results = np.zeros((folds,nb_seuils))
    X, y = CV_rep(Xtr, ytr, folds)

    for i in range(folds): 
        Xi = X[i]
        yi=y[i].to_numpy()
        yi[np.where(yi!=lab)[0]] = 3
        for j, s in enumerate(seuils):
            pred = 3*np.ones(yi.shape[0]).reshape(-1,1)
            index = np.where(Xi[n_var]>s)
            pred[index] = lab
            results[i,j] = f1(yi,pred, lab)
    return results.mean(axis=0)

def choix_seuils(X,y,folds,nb_seuils, n_var): 
    """Fonction qui choisi les seuils en fonction des résultats de cross validation

    Args:
        X (pd.DataFrame): Jeu de données avec les co-variables sur lesquel on veut entrainer les valeurs seuils
        y (pd.DataFrame): Jeu de données avec la variable à prédire sur lequel on veut entrainer les valeurs seuils
        folds (int): nombre de folds (sous-divisions) pour faire la cross-validation
        nb_seuils (int): nombre de seuils à choisir
        n_var (string): nom de la variable sur laquelle on travaille

    Returns:
        seuil_0 (float): seuil choisi pour distinguer les classes 1 et 0
        seuil_2 (float): seuil choisi pour distinguer les classes 0 et 2
    """
    vect_x = np.linspace(X[[n_var]].min()[0], X[[n_var]].max()[0], nb_seuils)
    res_0 = frontiere(X,y, 0, folds, nb_seuils, n_var)
    res_2 = frontiere(X,y, 2, folds, nb_seuils,n_var)
    plt.plot(vect_x, res_0, label="Scores classification 1 et (0-2)")
    plt.xlabel("Seuils correspondant aux valeurs de redshift")
    plt.ylabel("f1_score")
    plt.title("F1_score en fonction des seuils de classification")
    plt.plot(vect_x, res_2, label="Scores classification (1-0) et 2")
    #On prend les seuils qui maximisent la f1 
    seuil_0 = vect_x[np.where(res_0==res_0.max())[0][0]]
    seuil_2 = vect_x[np.where(res_2==res_2.max())[0][0]]
    return seuil_0, seuil_2

def predict(val0,val2, X_te, n_var): 
    """Crée notre prédiction en fonction des valeurs val0 et val2 calculées précédamment. 
    Il crée un vecteur prédiction avec des valeurs: 
    - 1: n_var appartenant à ]-inf,val0[
    - 0: n_var appartenant à [val0,val2[
    - 2: n_var appartenant à [val2,+inf[

    Args:
        val0 (float): valeur seuil calculée pour distinguer la classe 1 et 0
        val2 (float): valeur seuil calculée pour distinguer la classe 0 et 2
        X_te (pd.DataFrame): vecteur des covariables test sur lesquels on va regarder la valeur de n_var
        n_var (string): nom de la variable sur laquelle on base la prédiction

    Returns:
        pred: vecteur contenant les prédictions de notre méthode_seuil
    """
    pred = np.ones(X_te.shape[0])
    index_0 = np.where(X_te[n_var]>val0)[0]
    index_2 = np.where(X_te[n_var]>val2)[0]
    pred[index_0] = int(0)
    pred[index_2] = int(2)
    return pred.reshape(-1,1)

def tirage(p,pred1,pred2):
    """Fonction qui tire melange deux vecteurs de probabilités. Elle repere les indices dans lequels les deux predicitions 
    sont differentes et choisi une des deux valeurs en tirant aleatoirement une bernouilli avec proba p. 

    Args:
        p (np.float): probabilité associé au tirage aléatoire de la loi de Bernouilli
        pred1 (np.array): Vecteur contenant les prédictions de la méthode 1
        pred2 (np.array): Vecteur contenant les prédictions de la méthode 1

    Returns:
        res: nouveau vecteur de prédictions calculé à partir des deux autres prédictions 
    """
    res = pred1
    idx = np.where(pred1!=pred2)[0]
    for i in idx: 
        if np.random.binomial(1,p)==1: 
            res[i] = pred2[i]
    return res

def melange(folds,Xtr,ytr, var1, seuil1, model2, vars):
    """Fonction qui choisi la probabilité p (avec cross-validation) que l'on tirera sur une bernouilli pour mélanger deux prédictions

    Args:
        folds (int): nombre de 
        Xtr (np.array ou pd.DataFrame): Vecteur d'entrainement contenant les co-variables
        ytr (np.array ou pd.DataFrame): Vecteur d'entrainement contenant la variable a predire
        var1 (string): variable sur laquelle se basent les seuil1
        seuil1 (list): liste avec les deux seuils choisis avec la méthode basée sur var1 (de la forme [seuil_0,seuil_2])
        model2: modèle 2 entraîné

    Returns:
        proba: probabilité que l'on utilisera pour mélanger deux predictions
    """
    X, y = CV_rep(Xtr, ytr, folds)
    probas = np.linspace(0,1,20)
    results = np.zeros((folds,len(probas)))

    for i in range(folds): 
        Xr = X[i]
        yr=y[i].to_numpy()
        for j,p in enumerate(probas): 
            p1 = Label_Encode(yr, pd.DataFrame(predict(seuil1[0],seuil1[1], Xr, var1)))
            p2 = Label_Encode(yr, pd.DataFrame(model2.predict(Xr[vars])))
            pred = tirage(p,p1,p2)
            results[i,j] = f1_score(yr,pred, average="weighted")
    
    return probas[np.where(results.mean(axis=0)==results.mean(axis=0).max())[0][0]]

def pred_mel(pred1, pred2, p): 
    """Fonction qui mélange deux prédictions avec probabilité p sur une Bernouilli

    Args:
        pred1 (np.array): predicteur 1 (à mélanger)
        pred2 (np.array): prédicteur 2 (à mélanger)
        p (np.float): probabilité pour la Bernouilli

    Returns:
        tirage(p,p1,p2) (np.array): nouvelle prédiction contenant le mélange des deux prédicteurs
    """
    return tirage(p,pred1,pred2)