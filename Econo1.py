import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ==========================================
# ÉTAPE 1 : Chargement et Préparation des données
# ==========================================

# 1. Charger les données
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# 2. Création de la variable dépendante binaire (0 = Low/Medium, 1 = High)
# On utilise np.where : si la valeur est 'High', on met 1, sinon on met 0
df['Burnout_Risk_Binary'] = np.where(df['Burnout_Risk_Level'] == 'High', 1, 0)

# 3. Suppression des colonnes inutiles
# On enlève l'ID (qui n'a pas de pouvoir prédictif) et l'ancienne colonne cible
df = df.drop(columns=['Student_ID', 'Burnout_Risk_Level'])

# 4. Encodage des variables catégorielles (texte -> nombres)
# pd.get_dummies transforme les catégories en colonnes de 0 et 1 (One-Hot Encoding)
# drop_first=True permet d'éviter le piège de la colinéarité parfaite (utile pour le Logit)
df_encoded = pd.get_dummies(df, drop_first=True)

# Séparer les variables explicatives (X) de la variable à prédire (y)
X = df_encoded.drop(columns=['Burnout_Risk_Binary'])
y = df_encoded['Burnout_Risk_Binary']

# ==========================================
# ÉTAPE 2 : Partitionnement (Train / Test) et Mise à l'échelle
# ==========================================

# Division : 80% d'apprentissage (Train) et 20% de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation des données (moyenne = 0, écart-type = 1)
# Très important pour la régression logistique !
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# ÉTAPE 3 : Définition et Entraînement des Modèles
# ==========================================

# Initialisation des 3 modèles
# On utilise class_weight='balanced' car il y a moins de 'High' (1) que de 'Low/Medium' (0)
models = {
    "Modèle Logit (Régression Logistique)": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# ==========================================
# ÉTAPE 4 : Évaluation et Comparaison
# ==========================================

print("=== COMPARAISON DES PERFORMANCES DES MODÈLES (Sur l'échantillon de Test) ===\n")

for name, model in models.items():
    # 1. Entraînement du modèle sur les données d'apprentissage
    model.fit(X_train_scaled, y_train)
    
    # 2. Prédictions sur l'échantillon de test
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilités pour l'AUC
    
    # 3. Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 4. Affichage des résultats
    print(f"--- {name} ---")
    print(f"Accuracy (Précision globale) : {accuracy:.4f}")
    print(f"Precision (Précision des positifs) : {precision:.4f}")
    print(f"Recall (Sensibilité) : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC (Aire sous la courbe ROC) : {auc:.4f}")
    print(f"Matrice de confusion :\n{conf_matrix}\n")