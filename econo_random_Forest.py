import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Charger les données
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# 2. Définir X (les variables explicatives) et Y (la cible)
features = ['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 
            'Prompt_Engineering_Skill', 'Primary_Use_Case', 
            'Anxiety_Level_During_Exams', 'Paid_Subscription', 'Skill_Retention_Score']

X = df[features]
y = df['Post_Semester_GPA']

# 3. Encodage : Transformer le texte en chiffres (Obligatoire en Machine Learning)
X = pd.get_dummies(X, drop_first=True)

# 4. Séparer en Entraînement (80%) et Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Créer le modèle Random Forest (avec nos hyperparamètres)
foret = RandomForestRegressor(
    n_estimators=100,     # Le nombre d'arbres
    max_depth=10,         # La profondeur de chaque arbre
    n_jobs=-1,            # La vitesse d'exécution (utiliser tous les cœurs du PC)
    random_state=42
)

# 6. Entraîner le modèle sur les 80%
foret.fit(X_train, y_train)

# 7. Évaluer le modèle sur les 20% restants (les données "inconnues")
predictions = foret.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "="*45)
print("--- PERFORMANCES DE LA RANDOM FOREST ---")
print("="*45)
print(f"R² (sur le Test) : {r2:.4f}")
print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print("="*45 + "\n")