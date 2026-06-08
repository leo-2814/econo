import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# ÉTAPE 1 : Chargement et Préparation des données
# ==========================================

# 1. Charger les données
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# 2. Création de la variable dépendante binaire (0 = Low/Medium, 1 = High)
df['Burnout_Risk_Binary'] = np.where(df['Burnout_Risk_Level'] == 'High', 1, 0)

# 3. Suppression des colonnes inutiles
df = df.drop(columns=['Student_ID', 'Burnout_Risk_Level'])

# 4. Encodage des variables catégorielles (texte -> nombres)
df_encoded = pd.get_dummies(df, drop_first=True)

# Séparer les variables explicatives (X) de la variable à prédire (y)
X = df_encoded.drop(columns=['Burnout_Risk_Binary'])
y = df_encoded['Burnout_Risk_Binary']

# ==========================================
# ÉTAPE 2 : Partitionnement (Train / Test) et Mise à l'échelle
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation (Cruciale pour le SVM et le KNN !)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# ÉTAPE 3 : Définition et Entraînement des Modèles
# ==========================================


# --- DÉFINITION DES MODÈLES ---
models = {
    "Logit (Régression Logistique)": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    
    # Voici le SVM optimisé pour aller très vite sur 50 000 lignes sans message d'erreur :
    "Support Vector Machine (SVM)": CalibratedClassifierCV(LinearSVC(random_state=42, dual=False, max_iter=2000))
}

# ==========================================
# ÉTAPE 4 : Évaluation et Comparaison
# ==========================================

print("=== COMPARAISON DES PERFORMANCES DES 5 MODÈLES (Échantillon de Test) ===\n")

for name, model in models.items():
    # Entraînement
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Affichage
    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f} | F1-Score : {f1:.4f} | AUC : {auc:.4f}")
    print(f"Matrice de confusion :\n{conf_matrix}\n")
