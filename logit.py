import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
# Importation des modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
# Chargement de la base
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# Cible binaire : 1 = Risque de Burnout élevé
df['Burnout_Risk_Binary'] = np.where(df['Burnout_Risk_Level'] == 'High', 1, 0)
df = df.drop(columns=['Student_ID', 'Burnout_Risk_Level'])

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df, drop_first=True, dtype=float)

# Les variables validées du modèle optimisé
colonnes_a_garder = [
    'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
    'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams', 'Paid_Subscription',
    'Year_of_Study_Graduate', 'Year_of_Study_Junior', 'Year_of_Study_Senior',
    'Year_of_Study_Sophomore', 'Institutional_Policy_Strict_Ban', 
    'Primary_Use_Case_Direct_Answer_Generation'
]

# Définition des variables explicatives (X) et de la cible (y)
# On s'assure que seules les colonnes existantes après encodage sont conservées
colonnes_finales = [col for col in colonnes_a_garder if col in df_encoded.columns]
X = df_encoded[colonnes_finales]
y = df_encoded['Burnout_Risk_Binary']

# Séparation en jeu d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation des données (Indispensable pour KNN, SVM et Régression Logistique)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calcul dynamique du scale_pos_weight pour XGBoost (Ratio : nb_negatifs / nb_positifs)
ratio_classes = (y_train == 0).sum() / (y_train == 1).sum()

# ==========================================
# 2. FONCTION D'ÉVALUATION UNIVERSELLE
# ==========================================
def evaluer_modele(nom_modele, modele, X_test, y_test, utilise_proba=True):
    """Calcule et affiche l'Accuracy, la Précision, le F1-Score et l'AUC."""
    y_pred = modele.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"--- {nom_modele} ---")
    print(f"Accuracy  : {acc:.4f}")
    
    # Même si le SVM n'exigeait que l'Accuracy, afficher toutes les métriques 
    # permet de comparer les 4 modèles sur la même base.
    print(f"Precision : {prec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    
    # Calcul de l'AUC
    if utilise_proba:
        # Pour les modèles qui sortent des probabilités directes
        y_prob = modele.predict_proba(X_test)[:, 1]
    else:
        # Pour le SVM sans predict_proba
        y_prob = modele.decision_function(X_test)
        
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC       : {auc:.4f}\n")

# ==========================================
# 3. INITIALISATION ET ENTRAÎNEMENT DES MODÈLES
# ==========================================

# --- A. Régression Logistique ---
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_train_scaled, y_train)
evaluer_modele("Régression Logistique", logit_model, X_test_scaled, y_test)

# --- B. K-Nearest Neighbors (KNN) ---

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
evaluer_modele("K-Nearest Neighbors", knn_model, X_test_scaled, y_test)

# --- C. XGBoost ---
xgb_model = XGBClassifier(
    learning_rate=0.1,          # Vitesse d'apprentissage
    subsample=0.8,              # Sous-ensemble des données par arbre
    scale_pos_weight=ratio_classes, # Poids ajusté automatiquement selon vos données
    min_child_weight=1,         # Nombre d'obs minimum par feuille
    colsample_bytree=0.8,       # Nombre de variables par arbre
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)
# Note : on utilise l'argument utilise_proba=True par défaut
evaluer_modele("XGBoost", xgb_model, X_test_scaled, y_test)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

# --- D. Support Vector Machine (SVM) ---
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# On définit la grille de recherche pour le duo C et Sigma (gamma)
# C = Coût, gamma = Sigma
param_grid_svm = {
    'C': [1], 
    'gamma': [0.1, 'scale', 'auto'] 
}

grid_svm = GridSearchCV(
    estimator=SVC(kernel='rbf', random_state=42),
    param_grid=param_grid_svm,
    cv=3, # Réduit à 3 pour aller plus vite avec vos 40 000 lignes
    scoring='recall', 
    n_jobs=-1
)

grid_svm.fit(X_train_scaled, y_train)

print(f"Meilleur Coût (C) trouvé : {grid_svm.best_params_['C']}")
print(f"Meilleur Sigma (Gamma) trouvé : {grid_svm.best_params_['gamma']}")

# evaluer_modele("Support Vector Machine", svm_model, X_test_scaled, y_test)

# from sklearn.model_selection import RandomizedSearchCV

# # 1. On définit le modèle de base avec les paramètres FIXES
# # (On garde le scale_pos_weight car il dépend de vos données, pas de l'algorithme)
# xgb_base = XGBClassifier(
#     scale_pos_weight=ratio_classes, 
#     random_state=42,
#     eval_metric='logloss'
# )

# # 2. On définit les fourchettes à tester pour les paramètres VARIABLES
# espace_recherche_xgb = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Vitesse d'apprentissage
#     'subsample': [0.6, 0.8, 1.0],             # % d'étudiants par arbre
#     'colsample_bytree': [0.6, 0.8, 1.0],      # % de variables par arbre
#     'min_child_weight': [1, 3, 5, 7],         # Exigence pour créer une feuille
#     'max_depth': [3, 5, 7],                   # Profondeur maximale
#     'n_estimators': [100, 200, 300]           # Nombre d'arbres total
# }

# print("Lancement de la recherche (test de 20 combinaisons aléatoires)...")

# # 3. Configuration de la recherche aléatoire
# recherche_xgb = RandomizedSearchCV(
#     estimator=xgb_base,
#     param_distributions=espace_recherche_xgb,
#     n_iter=20,            # Le nombre de combinaisons à tester au hasard (modulable)
#     cv=5,                 # Validation croisée (fiabilité)
#     scoring='recall',     # On cherche toujours à maximiser la détection des burnouts
#     n_jobs=-1,            # Utilise toute la puissance de votre ordinateur
#     random_state=42,      # Pour pouvoir reproduire le même résultat
#     verbose=1             # Affiche l'avancement
# )

# # 4. Entraînement de l'algorithme de recherche
# recherche_xgb.fit(X_train_scaled, y_train)

# # 5. Affichage du résultat gagnant
# print("\n=== LES MEILLEURS PARAMÈTRES TROUVÉS SONT ===")
# for param, valeur in recherche_xgb.best_params_.items():
#     print(f"{param} : {valeur}")

# # 6. Évaluation automatique de ce super-modèle
# meilleur_xgb = recherche_xgb.best_estimator_
# evaluer_modele("XGBoost (Optimisé par RandomizedSearch)", meilleur_xgb, X_test_scaled, y_test)


# nb_negatifs = (y_train == 0).sum()
# nb_positifs = (y_train == 1).sum()
# ratio = nb_negatifs / nb_positifs

# print(f"Somme des réponses = 0 (négatifs) : {nb_negatifs}")
# print(f"Somme des réponses = 1 (positifs) : {nb_positifs}")
# print(f"La valeur de scale_pos_weight à inscrire est : {ratio:.2f}")