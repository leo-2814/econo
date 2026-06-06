# import pandas as pd
# import statsmodels.formula.api as smf

# # 1. Charger les données (Assure-toi que le fichier est dans le même dossier)
# df = pd.read_csv('ai_student_impact_dataset (1).csv')


# formule = "Post_Semester_GPA ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + C(Prompt_Engineering_Skill) + C(Primary_Use_Case) + Anxiety_Level_During_Exams+Paid_Subscription+Skill_Retention_Score"

# # 3. Créer et entraîner le modèle OLS (Moindres Carrés Ordinaires)
# modele_regression = smf.ols(formula=formule, data=df).fit()



# #4. Afficher le tableau de résultats complet
# print(modele_regression.summary())
# import pandas as pd
# import statsmodels.formula.api as smf

# # 1. Charger les données
# df = pd.read_csv('ai_student_impact_dataset (1).csv')

# # 2. Créer la variable binaire (1 pour High Burnout, 0 pour le reste)
# # .astype(int) transforme les "True/False" en "1/0"
# df['High_Burnout_Risk'] = (df['Burnout_Risk_Level'] == 'High').astype(int)

# # 3. Définir la formule du logit
# formule_logit = "High_Burnout_Risk ~ Pre_Semester_GPA + Weekly_GenAI_Hours + Traditional_Study_Hours + Anxiety_Level_During_Exams + Skill_Retention_Score + C(Paid_Subscription) + C(Prompt_Engineering_Skill) + C(Primary_Use_Case)"

# # 4. Entraîner le modèle Logit (on utilise smf.logit au lieu de smf.ols)
# modele_logit = smf.logit(formula=formule_logit, data=df).fit()

# # 5. Afficher les résultats
# print(modele_logit.summary())

import pandas as pd
import statsmodels.formula.api as smf

# 1. Charger les données
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# 2. Créer la variable binaire d'évolution (1 si progression, 0 si baisse/stagnation)
df['Grade_Improved'] = (df['Post_Semester_GPA'] > df['Pre_Semester_GPA']).astype(int)

# 3. Définir la formule du logit 
# (Note : Je n'ai pas mis Pre_Semester_GPA dans les explicatives pour voir le pur impact du travail et de l'IA)
formule_logit_evo = "Grade_Improved ~   Traditional_Study_Hours + Anxiety_Level_During_Exams + Skill_Retention_Score + C(Prompt_Engineering_Skill) + C(Primary_Use_Case)"

# 4. Entraîner le modèle Logit
modele_logit_evo = smf.logit(formula=formule_logit_evo, data=df).fit()

# 5. Afficher les résultats
print(modele_logit_evo.summary())