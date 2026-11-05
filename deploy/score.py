# score.py : script de scoring du modèle heart-model
# Azure ML appelle init() une seule fois au démarrage du conteneur,
# puis run() à chaque requête envoyée à l'API.

import os
import json
import pandas as pd
import mlflow.sklearn

model = None  # variable globale partagée entre init() et run()


def init():
    """Chargement du modèle au démarrage du conteneur."""
    global model
    # Azure ML copie le modèle enregistré dans le dossier AZUREML_MODEL_DIR
    racine = os.getenv("AZUREML_MODEL_DIR")

    # On cherche le sous-dossier qui contient le fichier MLmodel (format MLflow)
    chemin_modele = None
    for dossier, _, fichiers in os.walk(racine):
        if "MLmodel" in fichiers:
            chemin_modele = dossier
            break

    # Chargement du modèle scikit-learn sauvegardé par MLflow
    model = mlflow.sklearn.load_model(chemin_modele)
    print(f"[init] Modèle chargé depuis : {chemin_modele}")


def run(raw_data):
    """Appelée à chaque requête. raw_data est le texte JSON envoyé à l'API."""
    try:
        # 1. Lecture du JSON : {"data": [ {colonne: valeur, ...}, ... ]}
        corps = json.loads(raw_data)
        if not isinstance(corps, dict) or "data" not in corps:
            print("[run] ERREUR : clé data absente")
            return {"erreur": 'Format attendu : {"data": [{colonne: valeur, ...}]}'}

        # 2. Conversion en DataFrame, valeurs en décimaux comme à l'entraînement
        X = pd.DataFrame(corps["data"]).astype(float)

        # 3. On remet les colonnes dans l'ordre vu à l'entraînement
        colonnes = getattr(model, "feature_names_in_", None)
        if colonnes is not None:
            manquantes = [c for c in colonnes if c not in X.columns]
            if manquantes:
                print(f"[run] ERREUR : colonnes manquantes {manquantes}")
                return {"erreur": f"{len(manquantes)} colonne(s) manquante(s) : {manquantes}"}
            X = X[list(colonnes)]

        # 4. Prédiction (0 = survie, 1 = décès) et probabilité de décès
        predictions = model.predict(X).tolist()
        probabilites = model.predict_proba(X)[:, 1].round(3).tolist()
        print(f"[run] {len(X)} ligne(s) reçue(s), prédictions : {predictions}")
        return {"predictions": predictions, "probabilites_deces": probabilites}

    except Exception as erreur:
        # En cas de problème, on renvoie un message lisible au lieu de planter
        print(f"[run] ERREUR : {erreur}")
        return {"erreur": str(erreur)}
