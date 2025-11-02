
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import predictor

st.set_page_config(page_title="IA Santé – Prédiction avancée", page_icon="🩺", layout="wide")

st.image("logo_app.png", use_column_width=False, width=420)
st.title("🩺 IA Santé – Prédiction du risque (version avancée)")
st.caption("Stress (1–5) • ICS (0–4) • SCG (0–10) • Âge • Poste de nuit → Probabilité & Classe")

# Load model (if present)
model = predictor.load_model()
if model is not None:
    st.success("Modèle scikit-learn chargé ✅")
else:
    st.info("Aucun modèle trouvé. Utilisation de la formule logistique interprétable.")

seuil = st.slider("Seuil de classification (Stable / À risque)", 0.30, 0.70, 0.50, 0.01)

tab1, tab2, tab3 = st.tabs(["🔹 Prédiction unitaire", "📁 Prédiction par CSV", "ℹ️ À propos"])

with tab1:
    st.subheader("Prédiction unitaire")
    c1, c2, c3 = st.columns(3)
    with c1:
        stress = st.slider("Stress (1–5)", 1.0, 5.0, 3.0, 0.1)
        ics = st.slider("ICS (0–4)", 0, 4, 1, 1)
    with c2:
        scg = st.slider("SCG (0–10)", 0.0, 10.0, 7.0, 0.1)
        age = st.number_input("Âge", 18, 75, 40)
    with c3:
        poste_nuit = st.selectbox("Poste de nuit", ["Non","Oui"])
        btn = st.button("Prédire", use_container_width=True)

    if btn:
        row = pd.DataFrame([{
            "id":"UNITE",
            "stress": stress, "ics": ics, "scg": scg, "age": age,
            "poste_nuit": 1 if poste_nuit=="Oui" else 0
        }])
        errs = predictor.validate(row.copy())
        if errs:
            for e in errs:
                st.error(e)
        else:
            out = predictor.predict_df(row, model=model, seuil=seuil)
            prob = float(out.loc[0,"probabilite"])
            classe = out.loc[0,"classe"]
            m1, m2 = st.columns(2)
            m1.metric("Probabilité de risque", f"{prob:.1f}%")
            m2.metric("Classe", classe)
            # importances (coeffs) si dispo
            imps = predictor.coef_importances(model) if model is not None else []
            if imps:
                st.write("**Importance (coefficients du modèle)**")
                feats = [i[0] for i in imps]
                vals = [abs(i[1]) for i in imps]
                fig, ax = plt.subplots()
                ax.barh(feats[::-1], vals[::-1])
                ax.set_xlabel("Importance (|coefficient|)")
                st.pyplot(fig)

with tab2:
    st.subheader("Prédiction par CSV")
    st.caption("Colonnes obligatoires : id, stress, ics, scg, age, poste_nuit (0/1)")
    file = st.file_uploader("Charger un CSV", type=["csv"])
    calc_shap = st.checkbox("Calculer explications SHAP (démo, max 20 lignes)", value=False)
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write("Aperçu :")
            st.dataframe(df.head())
            errs = predictor.validate(df.copy())
            if errs:
                for e in errs:
                    st.error(e)
            else:
                out = predictor.predict_df(df.copy(), model=model, seuil=seuil)
                st.success("Prédictions effectuées ✅")
                st.dataframe(out)
                # KPIs
                st.subheader("Résumé")
                kpi = predictor.resume_df(out)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total", kpi["Total"])
                k2.metric("% À risque", kpi["% À risque"])
                k3.metric("Stress moyen", kpi["Stress moyen"])
                k4.metric("SCG moyen", kpi["SCG moyen"])
                # Classes chart
                st.subheader("Répartition des classes")
                counts = out["classe"].value_counts().reindex(["Stable","À risque"]).fillna(0)
                fig, ax = plt.subplots()
                ax.bar(counts.index, counts.values)
                ax.set_ylabel("Effectif")
                st.pyplot(fig)
                # Download
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Télécharger résultats (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
                # SHAP (optionnel)
                if calc_shap and (model is not None):
                    try:
                        st.subheader("Explications SHAP (démo)")
                        X = df[["stress","ics","scg","age","poste_nuit"]].copy()
                        sv, X_eval = predictor.shap_values_for_batch(model, X, max_background=20, max_rows=20)
                        if sv is not None:
                            import shap
                            fig2 = shap.plots.beeswarm(sv, show=False)
                            st.pyplot(fig2)
                        else:
                            st.info("Pas de données suffisantes pour SHAP.")
                    except Exception as e:
                        st.warning(f"SHAP non disponible: {e}")
        except Exception as e:
            st.error(f"Erreur de lecture: {e}")

with tab3:
    st.markdown("""**Variables** :
- **Stress (1–5)** : plus élevé → risque ↑
- **ICS (0–4)** : facteurs cardio (HTA, diabète, tabac, ATCD) → risque ↑
- **SCG (0–10)** : cognition (attention, vigilance) → protecteur si élevé
- **Âge** et **Poste de nuit** : facteurs contextuels

**Avertissement** : Cet outil est une aide à la décision, il **ne remplace pas** le jugement clinique.
""")
