import pandas as pd
import statsmodels.formula.api as smf

# 1. Charger les données (Assure-toi que le fichier est dans le même dossier)
df = pd.read_csv('ai_student_impact_dataset (1).csv')


formule = "Post_Semester_GPA ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams+Paid_Subscription+Skill_Retention_Score"

# 3. Créer et entraîner le modèle OLS (Moindres Carrés Ordinaires)
modele_regression = smf.ols(formula=formule, data=df).fit()



#4. Afficher le tableau de résultats complet
print(modele_regression.summary())

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Charger les données (Assure-toi que le fichier est dans le même dossier)
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# 2. Définir la formule
formule = "Post_Semester_GPA ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams+Paid_Subscription+Skill_Retention_Score"

# 3. Créer et entraîner le modèle OLS (Moindres Carrés Ordinaires)
modele_regression = smf.ols(formula=formule, data=df).fit()

# 4. Afficher le tableau de résultats complet de la régression
print(modele_regression.summary())

# ==========================================
# --- 5. CALCUL DES MÉTRIQUES D'ERREUR ---
# ==========================================

# R² Ajusté (déjà contenu dans le modèle)
r2_ajuste = modele_regression.rsquared_adj

# On récupère les "Vraies Notes" de la base de données
valeurs_reelles = df['Post_Semester_GPA']

# On demande au modèle de "deviner" les notes pour voir s'il est précis
predictions = modele_regression.predict(df)

# Calcul du RMSE et MAE via scikit-learn
rmse = np.sqrt(mean_squared_error(valeurs_reelles, predictions))
mae = mean_absolute_error(valeurs_reelles, predictions)

print("\n" + "="*40)
print("--- PERFORMANCES DU MODÈLE OLS ---")
print("="*40)
print(f"R² Ajusté : {r2_ajuste:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")
print("="*40 + "\n")


import pandas as pd
import statsmodels.formula.api as smf

# 1. Charger les données
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# 2. La formule classique
formule = "Post_Semester_GPA ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams + Paid_Subscription + Skill_Retention_Score"

# 3. LE CORRECTIF MAGIQUE : Ajouter cov_type='HC3' pour corriger l'hétéroscédasticité
modele_robuste = smf.ols(formula=formule, data=df).fit(cov_type='HC3')

# 4. Afficher le nouveau tableau inattaquable
print(modele_robuste.summary())



# import pandas as pd
# import numpy as np
# import statsmodels.formula.api as smf
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # 1. Charger les données
# df = pd.read_csv('ai_student_impact_dataset (1).csv')

# # 2. Créer les variables logarithmiques (on ajoute +1 pour éviter le log de 0)
# df['Log_GenAI_Hours'] = np.log(df['Weekly_GenAI_Hours'] + 1)
# df['Log_Study_Hours'] = np.log(df['Traditional_Study_Hours'] + 1)

# # 3. Définir la nouvelle formule (Modèle Lin-Log)
# # On a remplacé les heures brutes par nos nouvelles variables Log
# formule_log = "Post_Semester_GPA ~ Pre_Semester_GPA + Log_GenAI_Hours + Log_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams + Paid_Subscription + Skill_Retention_Score"

# # 4. Entraîner le modèle OLS avec les logarithmes
# modele_log = smf.ols(formula=formule_log, data=df).fit()
# print(modele_log.summary())

# # ==========================================
# # --- 5. CALCUL DES MÉTRIQUES D'ERREUR ---
# # ==========================================

# r2_ajuste_log = modele_log.rsquared_adj
# valeurs_reelles = df['Post_Semester_GPA']
# predictions_log = modele_log.predict(df)

# rmse_log = np.sqrt(mean_squared_error(valeurs_reelles, predictions_log))
# mae_log = mean_absolute_error(valeurs_reelles, predictions_log)

# print("\n" + "="*50)
# print("--- PERFORMANCES DU MODÈLE LOGARITHMIQUE ---")
# print("="*50)
# print(f"R² Ajusté : {r2_ajuste_log:.4f}")
# print(f"RMSE      : {rmse_log:.4f}")
# print(f"MAE       : {mae_log:.4f}")
# print("="*50 + "\n")