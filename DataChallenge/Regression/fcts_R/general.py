from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
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

def param_selection(param, mod, Xtr, ytr, Xte): 
    """_summary_

    Args:
        param (liste):  {"param1":['option1', 'option2'],"param2":[option1,option2,option3]}
        mod (sklearn.function): model(random_state=10)
        Xtr (pd.DataFrame): Co-variables du jeu de données train 
        ytr (pd.DataFrame): Variable à prédire du jeu de données train 
        Xte (pd.DataFrame): Co-variables du jeu de données test

    Returns:
        pred (np.array ou pd.DataFrame): vecteur contenant les prédictions du modèle selectionné par cross validation
    """

    grid_search = GridSearchCV(estimator=mod,param_grid=param,cv=5)
    grid_search.fit(Xtr, ytr)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    pred = grid_search.predict(Xte)

    return pred 


def train_eval(model, X, y, X_test, y_test):
    """Fonction qui entraine un modèle, plotte et renvoit le r2_score sur le jeu de données test associé à la prédiction

    Args:
        model (sklearn.function): Modèle à entrainer pour faire la prediction
        X (pd.Dataframe): Co-variables du jeu de données train
        y (_type_): Variable à prédire du jeu de données train
        X_test (_type_): Co-variables du jeu de données test
        y_test (_type_): Variable à prédire du jeu de données test

    Returns:
        pred (np.array ou pd.DataFrame): prédiction sur le jeu de données test associé au modèle choisi en input
    """
    lab = str(model()) 
    #Entrainement du modèle 
    mod = model()
    mod.fit(X,y)
    pred = mod.predict(X_test)

    plt.figure()
    plt.hist(y_test, density=True, label="True", alpha=0.5, bins=np.linspace(3,9,7))
    plt.hist(pred, density=True, label=lab, alpha=0.5, bins=np.linspace(3,9,7))
    plt.title("Histogramme des predicitions de "+lab)
    plt.legend()
    plt.show()

    print("normal: ", r2_score(y_test, pred))
    return pred


def build_pred(X_test0, X_test1, pred0,pred1): 
    """Fonction qui combine les prédictions de data0 et data1 

    Args:
        X_test0 (pd.DataFrame): Vecteur contenant les co-variables à wine_type=0 du jeu de données test 
        X_test1 (pd.DataFrame): Vecteur contenant les co-variables à wine_type=0 du jeu de données test 
        pred0 (np.array ou pd.DataFrame): prédictions associés pour les indices de wine_type=0
        pred1 (np.array ou pd.DataFrame): prédictions associés pour les indices de wine_type=1

    Returns:
        pred (np.array): Vecteur contenant l'ensemble des prédictions du jeu de données test 
    """
    data_0 = np.column_stack((X_test0[["wine_ID"]].to_numpy(), pred0))
    data_1 = np.column_stack((X_test1[["wine_ID"]].to_numpy(), pred1))
    pred = np.row_stack((data_0,data_1))
    return pred 

def soumission(pred, date, name_pred): 
    """Fonction qui crée une soumission avec le nom de la methode et la sauvagarde dans datasets_c

    Args:
        pred (np.array): prediction 
        date (str): date ex: '0110'
        name_pred (str): nom de la méthode utilisée
    """
    # Nommez les colonnes
    column_names = ['wine_ID', 'target']
    # Spécifiez le nom du fichier CSV de sortie
    csv_filename = 'submissions_R/'+name_pred+'_'+date+'.csv'
    # Ouvrez le fichier CSV en mode écriture et écrivez les données
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Écrivez les noms des colonnes
        writer.writerow(column_names)
        # Écrivez les données
        writer.writerows(pred)
    return print('OK')