import pandas as pd
import numpy as np
import time

# Outils StatsModels (Économétrie)
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Outils Scikit-Learn (Machine Learning)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

# Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==============================================================================
# 0. PRÉPARATION GLOBALE DES DONNÉES
# ==============================================================================
print("Chargement et préparation des données en cours...")
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# --- PRÉPARATION POUR LA RÉGRESSION (GPA) ---
colonnes_gpa = [
    'Post_Semester_GPA', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 
    'Traditional_Study_Hours', 'Prompt_Engineering_Skill', 
    'Primary_Use_Case', 'Anxiety_Level_During_Exams', 
    'Paid_Subscription', 'Skill_Retention_Score'
]
df_gpa = df.dropna(subset=colonnes_gpa).copy()

# Séparation Train/Test pour statsmodels (RLM)
df_train_gpa, df_test_gpa = train_test_split(df_gpa, test_size=0.2, random_state=42)

# Préparation X et y pour Scikit-Learn (RF & GBR)
X_reg = df_gpa.drop(columns=['Post_Semester_GPA'])
y_reg = df_gpa['Post_Semester_GPA']
X_reg_encoded = pd.get_dummies(X_reg, drop_first=True, dtype=float)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=42)

# --- PRÉPARATION POUR LA CLASSIFICATION (BURNOUT) ---
# Cible binaire : 1 = Risque de Burnout élevé
df['Burnout_Risk_Binary'] = np.where(df['Burnout_Risk_Level'] == 'High', 1, 0)
df_class = df.drop(columns=['Student_ID', 'Burnout_Risk_Level']).copy()
df_class_encoded = pd.get_dummies(df_class, drop_first=True, dtype=float)

colonnes_a_garder_class = [
    'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
    'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams', 'Paid_Subscription',
    'Year_of_Study_Graduate', 'Year_of_Study_Junior', 'Year_of_Study_Senior',
    'Year_of_Study_Sophomore', 'Institutional_Policy_Strict_Ban', 
    'Primary_Use_Case_Direct_Answer_Generation'
]
colonnes_finales_class = [col for col in colonnes_a_garder_class if col in df_class_encoded.columns]

X_class = df_class_encoded[colonnes_finales_class]
y_class = df_class_encoded['Burnout_Risk_Binary']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Standardisation pour la classification (Logit, KNN, SVM)
scaler = StandardScaler()
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

ratio_classes = (y_train_class == 0).sum() / (y_train_class == 1).sum()
print("✅ Préparation terminée.\n")


# ==============================================================================
# PARTIE 1 : VARIABLE CONTINUE (GPA)
# Structure : RLM -> Random Forest -> Gradient Boost
# ==============================================================================
print("="*60)
print(" PARTIE 1 : PRÉDICTION DU GPA (RÉGRESSION)")
print("="*60)

# --- 1. RLM (Régression Linéaire Multiple) ---
print("\n--- 1. RÉGRESSION LINÉAIRE MULTIPLE (OLS ROBUSTE) ---")
formule = "Post_Semester_GPA ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams + Paid_Subscription + Skill_Retention_Score"

# Entraînement avec correction de l'hétéroscédasticité (écarts-types robustes HC3)
modele_regression = smf.ols(formula=formule, data=df_train_gpa).fit(cov_type='HC3')

# 1. Le résumé complet de la régression (Indispensable en économétrie)
print(modele_regression.summary())

# 2. Les performances hors échantillon
predictions_test_ols = modele_regression.predict(df_test_gpa)
print("\n> PERFORMANCES HORS ÉCHANTILLON (TEST) :")
print(f"R² (Test) : {r2_score(df_test_gpa['Post_Semester_GPA'], predictions_test_ols):.4f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(df_test_gpa['Post_Semester_GPA'], predictions_test_ols)):.4f}")
print(f"MAE       : {mean_absolute_error(df_test_gpa['Post_Semester_GPA'], predictions_test_ols):.4f}")


# --- 2. Random Forest ---
print("\n--- 2. RANDOM FOREST REGRESSOR ---")
foret = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, max_features=10)
foret.fit(X_train_reg, y_train_reg)
predictions_rf = foret.predict(X_test_reg)

print(f"R² (Test) : {r2_score(y_test_reg, predictions_rf):.4f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test_reg, predictions_rf)):.4f}")
print(f"MAE       : {mean_absolute_error(y_test_reg, predictions_rf):.4f}")


# --- 3. Gradient Boost ---
print("\n--- 3. GRADIENT BOOSTING REGRESSOR ---")
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train_reg, y_train_reg)
predictions_gbr = gbr.predict(X_test_reg)

print(f"R² (Test) : {r2_score(y_test_reg, predictions_gbr):.4f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test_reg, predictions_gbr)):.4f}")
print(f"MAE       : {mean_absolute_error(y_test_reg, predictions_gbr):.4f}")


# ==============================================================================
# PARTIE 2 : VARIABLE BINAIRE (BURNOUT)
# Structure : Logit -> K-Nearest -> XGBoost -> Vector (SVM)
# ==============================================================================
print("\n\n" + "="*60)
print(" PARTIE 2 : PRÉDICTION DU BURNOUT (CLASSIFICATION)")
print("="*60)

def evaluer_classif(nom_modele, modele, X_test, y_test, utilise_proba=True):
    y_pred = modele.predict(X_test)
    if utilise_proba:
        y_prob = modele.predict_proba(X_test)[:, 1]
    else:
        y_prob = modele.decision_function(X_test)
        
    print(f"\n--- {nom_modele.upper()} ---")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"AUC       : {roc_auc_score(y_test, y_prob):.4f}")


# --- 1. Logit (Régression Logistique) ---
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_train_class_scaled, y_train_class)
evaluer_classif("1. Régression Logistique (Logit)", logit_model, X_test_class_scaled, y_test_class)


# --- 2. K-Nearest Neighbors ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_class_scaled, y_train_class)
evaluer_classif("2. K-Nearest Neighbors (KNN)", knn_model, X_test_class_scaled, y_test_class)


# --- 3. XGBoost ---
xgb_model = XGBClassifier(
    learning_rate=0.1, subsample=0.8, scale_pos_weight=ratio_classes, 
    min_child_weight=1, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)
xgb_model.fit(X_train_class_scaled, y_train_class)
evaluer_classif("3. XGBoost", xgb_model, X_test_class_scaled, y_test_class)


# --- 4. Vector (Support Vector Machine) ---
print("\n--- 4. SUPPORT VECTOR MACHINE (SVM) ---")
print("(Entraînement en cours sur un sous-échantillon de 5000 lignes pour optimiser le temps de calcul...)")
X_train_svm = X_train_class_scaled[:5000]
y_train_svm = y_train_class[:5000]

svm_final = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42, probability=True)
svm_final.fit(X_train_svm, y_train_svm)

evaluer_classif("Support Vector Machine", svm_final, X_test_class_scaled, y_test_class)

print("\n✅ Script complet terminé avec succès.")