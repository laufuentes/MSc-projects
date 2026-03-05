import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

folds = 5

def choix_var_knn(X_tr, y_tr, vars, columns): 
    """Fonction qui calcule pour un vecteur de variables, le f1_score (par cross-validation) associé à l'ajout d'une nouvelle variable.  
    Cette dernière choisit une nouvelle variable parmi cols et la rajoute sur vars.  

    Args:
        X_tr (pd.DataFrame): Jeu de données d'entraînement avec les co-variables
        y_tr (pd.DataFrame ou np.array): Jeu de données d'entraînement avec la variable à prédire 
        vars (pd.Index): vecteur avec les variables déjà selectionnées auparavant 
        columns (pd.Index): vecteur avec des nouvelles variables non inclues dans vars 

    Returns:
        vars (pd.Index): retourne le vecteurs vars avec la nouvelle variable maxiimisant le f1_score 
        cols (pd.Index): retourne le vecteur de nouvelles co-variables sans la variable finalement choisie
    """
    res = np.zeros(len(columns))
    for i, var in enumerate(columns): 
        xvar = vars.append(pd.Index([var], dtype='object')) 
        res[i] = np.mean(cross_validate(KNeighborsClassifier(), X_tr[xvar].to_numpy(), y_tr.to_numpy().ravel(), cv=folds,return_estimator=True)['test_score'])
    variable = columns[np.where(res==res.max())]
    return vars.append(pd.Index(variable, dtype='object')), columns.drop(variable)

def pred_acc_var(vars, X_tr,y_tr,X_te, y_te): 
    """Fonction qui pour des variables données, enmtraîne une knn et calcule l'erreur test associé. 

    Args:
        vars (pd.Index): variables selectionnées
        X_tr (pd.DataFrame): Jeu de données d'entraînement avec les co-variables
        y_tr (pd.DataFrame ou np.array): Jeu de données d'entraînement avec la variable à prédire
        X_te (pd.DataFrame): Jeu de données test avec les co-variables
        y_te (pd.DataFrame ou np.array): Jeu de données test avec la variable à prédire

    Returns:
        f1_score(y_te, pred, average="weighted"): score f1 sur le jeu de données test
    """
    knn = KNeighborsClassifier()
    knn.fit(X_tr[vars], y_tr.to_numpy().ravel())
    pred = knn.predict(X_te[vars])
    return f1_score(y_te, pred, average="weighted")

def plot_vars(res_plot, vars): 
    """Fonction qui renvoit un plot avec les f1_scores associés aux prédicteurs issus de l'ajout itératif de chaque variable

    Args:
        res_plot (list): liste contenant les scores f1 associés à l'ajout de chaque variable  
        vars (pd.Index): vecteur contenant les variables ajoutées dans l'ordre 
    """
    plt.plot(vars, res_plot)
    plt.xlabel("Ajout itératif des covariables")
    plt.ylabel("f1_score")
    plt.title("Selection de variables")

def main(X_tr, y_tr, X_te, y_te): 
    """Fonction qui effectue la selection de variables

    Args:
        X_tr (pd.DataFrame): Jeu de données d'entraînement avec les co-variables
        y_tr (pd.DataFrame ou np.array): Jeu de données d'entraînement avec la variable à prédire
        X_te (pd.DataFrame): Jeu de données test avec les co-variables
        y_te (pd.DataFrame ou np.array): Jeu de données tets avec la variable à prédire

    Returns:
        choix_vars (pd.Index): vecteur contenant le choix des variables aménant au meilleur f1_score par cross validation 
    """
    #Initialisation 
    vars = pd.Index([],dtype='object') #aucun choix de variables initialement 
    cols = X_tr.columns #ensemble des covariables
    res_plot = [] #on crée le vecteur dans lequel on sauvegardera itérativement le score associé à la nouvelle variable ajoutée

    while len(cols)>0: 
        vars, cols = choix_var_knn(X_tr,y_tr, vars,cols) #on ajoute la nouvelle variable qui maximise le f1_score 
        res_plot.append(pred_acc_var(vars, X_tr,y_tr,X_te, y_te)) #calcul du f1_score sur le jeu de données test 
    
    plot_vars(res_plot, vars) #figure
    choix_vars = vars[0:np.argmax(res_plot)+1]
    return choix_vars

def train(X,y,X_test, vars): 
    """Fonction qui entraîne le modèle pour les variables choisies et crée une prédiction  

    Args:
        X (pd.DataFrame): Jeu de données d'entraînement avec les co-variables
        y (pd.DataFrame ou np.array): Jeu de données d'entraînement avec la variable à prédire
        X_test (pdDataFrame): Jeu de données test avec les co-variables
        vars (pd.Index): vecteur contenant les variables choisies au préalable 

    Returns:
        _type_: _description_
    """
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X[vars],y.to_numpy().ravel())
    pred = knn.predict(X_test[vars])
    return pred 