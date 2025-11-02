
# IA Santé – Application de prédiction (version avancée avec SHAP + logo)

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Streamlit Cloud
1. Poussez ces fichiers sur un dépôt GitHub.
2. Allez sur https://share.streamlit.io → **New app**.
3. Sélectionnez votre dépôt, branche et `app.py`, puis **Deploy**.

## Utilisation
- **Onglet Prédiction unitaire** : curseurs + seuil → probabilité & classe.
- **Onglet CSV** : chargez `exemple_donnees.csv` (ou votre fichier) → prédictions, résumé, export CSV.
- Option **SHAP** : explication visuelle (démo, max 20 lignes).

## Modèle
- Si `modele_risque.joblib` est présent → utilisé automatiquement (pipeline scikit-learn).
- Sinon → **formule logistique interprétable** comme secours.
