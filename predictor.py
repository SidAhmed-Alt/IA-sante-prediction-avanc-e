
import numpy as np
import pandas as pd
import joblib

REQUIRED = ["id","stress","ics","scg","age","poste_nuit"]

def load_model(path="modele_risque.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def validate(df: pd.DataFrame):
    errs = []
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        errs.append(f"Colonnes manquantes: {miss}")
        return errs
    # Types
    try:
        df["stress"] = df["stress"].astype(float)
        df["ics"] = df["ics"].astype(int)
        df["scg"] = df["scg"].astype(float)
        df["age"] = df["age"].astype(int)
        df["poste_nuit"] = df["poste_nuit"].astype(int)
    except Exception:
        errs.append("Types invalides: stress/scg en float, ics/age/poste_nuit en entier (poste_nuit=0/1)")
    # Bornes
    if "stress" in df and ((df["stress"]<1) | (df["stress"]>5)).any():
        errs.append("stress hors bornes [1..5]")
    if "ics" in df and (~df["ics"].isin([0,1,2,3,4])).any():
        errs.append("ics doit être 0..4")
    if "scg" in df and ((df["scg"]<0) | (df["scg"]>10)).any():
        errs.append("scg hors bornes [0..10]")
    if "poste_nuit" in df and (~df["poste_nuit"].isin([0,1])).any():
        errs.append("poste_nuit doit être 0 ou 1")
    return errs

def proba_secours(stress, ics, scg, age, poste_nuit):
    z = 0.9*stress + 0.7*ics - 0.45*scg + 0.015*age + 0.35*poste_nuit - 1.5
    return 1/(1+np.exp(-z))

def predict_df(df: pd.DataFrame, model=None, seuil=0.5):
    X = df["stress ics scg age poste_nuit".split()].copy()
    if model is not None:
        proba = model.predict_proba(X)[:,1]
    else:
        proba = np.array([proba_secours(*row) for row in X.to_numpy()])
    classe = np.where(proba>=seuil, "À risque", "Stable")
    out = df.copy()
    out["probabilite"] = (proba*100).round(1)
    out["classe"] = classe
    return out

def resume_df(out: pd.DataFrame):
    total = len(out)
    pct_risk = float(out["classe"].eq("À risque").mean()) if total>0 else 0.0
    return {
        "Total": total,
        "% À risque": f"{round(pct_risk*100,1)}%",
        "Stress moyen": round(float(out["stress"].mean()),2) if total>0 else 0.0,
        "ICS moyen": round(float(out["ics"].mean()),2) if total>0 else 0.0,
        "SCG moyen": round(float(out["scg"].mean()),2) if total>0 else 0.0,
    }

def coef_importances(model):
    try:
        clf = model.named_steps.get("clf", None)
        if clf is None:
            return []
        coefs = clf.coef_[0]
        feats = ["stress","ics","scg","age","poste_nuit"]
        imp = sorted(zip(feats, coefs), key=lambda x: abs(x[1]), reverse=True)
        return [(f, float(c)) for f,c in imp]
    except Exception:
        return []

def shap_values_for_batch(model, X, max_background=20, max_rows=20):
    import shap
    if len(X) == 0:
        return None, None
    X_bg = X.iloc[:max_background, :]
    X_eval = X.iloc[:max_rows, :]
    explainer = shap.Explainer(model.predict_proba, X_bg, feature_names=X.columns)
    sv = explainer(X_eval)
    return sv, X_eval
