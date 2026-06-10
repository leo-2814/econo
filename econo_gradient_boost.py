import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Charger les données
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# 2. Définir X (les explicatives) et Y (la cible : la note finale)
features = ['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 
            'Prompt_Engineering_Skill', 'Primary_Use_Case', 
            'Anxiety_Level_During_Exams', 'Paid_Subscription', 'Skill_Retention_Score']

X = df[features]
y = df['Post_Semester_GPA']

# 3. Encodage du texte en nombres
X = pd.get_dummies(X, drop_first=True)

# 4. Séparation Train/Test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Créer le modèle Gradient Boosting
# ATTENTION : Les hyperparamètres sont différents d'une Random Forest !
gbr = GradientBoostingRegressor(
    n_estimators=100,      # Nombre d'étapes de correction
    learning_rate=0.1,     # La vitesse à laquelle il apprend (très important !)
    max_depth=3,           # Profondeur des arbres (on fait exprès de faire des petits arbres)
    random_state=42
)

# 6. Entraîner le modèle
gbr.fit(X_train, y_train)

# 7. Évaluer le modèle sur les données inconnues
predictions = gbr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "="*45)
print("--- PERFORMANCES DU GRADIENT BOOSTING ---")
print("="*45)
print(f"R² (sur le Test) : {r2:.4f}")
print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print("="*45 + "\n")