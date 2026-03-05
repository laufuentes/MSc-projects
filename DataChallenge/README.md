# Méthodes d'apprentissage supervisé: Data Challenge 
Benali Nafissa, Fuentes Vicente Laura 

M2 Mathématiques et Intelligence Artificielle 2023/2024 

## Introduction: 
Bienvenue sur notre repositoire GitHub pour Méthodes d'Apprentissage Supervisé et Data Challenge. Dans le cadre du Data Challenge nous avons été amenées à construire des modèles prédictifs pour deux problèmes différents, un de classification et un de régréssion: 
- Classification: "Classification d'objets célestes"
- Régréssion: "Prédiction de la qualité d'un vin"

Pour organiser ce repositoire, nous avons décidé de créer deux dossiers (Regression et Classification). Dans chacun de ces dossiers, vous pourrez trouver les notebooks python montrant notre démarche amenant au choix de chacun des modèles de prédiction finaux.  

## Guide du repositoire:
**0- Etape préliminaire: Création d'un environnement conda:**
Pour éviter des problèmes, nous proposons de créer un environnement Conda adapté à nos besoins. Celui-ci servira ultérieurement comme kernel pour les notebooks. Pour cela, il suffit d'exécuter dans le terminal (au niveau de notre dépôt), les commandes suivantes :
```
conda create --name <env_name> --file env_requirements.txt
```

**1- Localisation des datasets train et test**
Pour pouvoir implémenter les notebooks, il faudra préciser la localisation des datasets pour le problème de regression et classification. 
Pour cela, in faudra compléter l'entête du premier notebook de chacun des problèmes, en précisant les chemins vers les datasets train et test. 

**2- Implémentation des notebooks**
Pour chacun des problèmes, on pourra suivre les étapes de notre démarche en implémentant les notebooks de chaque dossier dans l'ordre. 

**3- Fonctions crées**
Pour faire face aux deux problèmes, nous avons crée des fonctions spécialisés qui se trouvent sur les sous-dossiers: fcts_r (resp. fcts_c) pour la régréssion (resp. classification). Vous pourrez trouver la documentation des fonctions sur les pdf à l'intérieur de chacun des sous-dossier. 

*NB: Pour pouvoir lancer les notebooks, il faudra choisir l'environnement conda crée précédemment en tant que kernel.*

