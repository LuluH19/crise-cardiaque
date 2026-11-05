"""
train.py
Entraînement du modèle de prédiction du décès (DEATH_EVENT), jeu Heart Failure.
Repris du TP 5 : MLflow + validation et enregistrement conditionnel du modèle.

Changements par rapport au TP :
  - lecture sans index_col=0 : ce jeu n'a pas de colonne d'identifiant ;
  - cible DEATH_EVENT, test_size de 0.2 (environ 60 patients, voir la fiche A) ;
  - l'enregistrement n'a lieu qu'avec --register : les jobs de comparaison des
    étapes 3 et 4 n'ajoutent donc aucune version au registre ;
  - le tag validation est aussi posé sur la version du modèle.

Exécutable :
  - en local : python src/train.py --data <chemin>
  - à distance sur un cluster, via un job Azure ML (command)
"""
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "DEATH_EVENT"


def parse_args():
    """1. Lecture des paramètres passés au script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Chemin du fichier CSV (data asset heart-failure-data)")
    parser.add_argument("--class_weight", type=str, default="none",
                        choices=["none", "balanced"],
                        help="Pondération des classes")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Part des données gardée pour le test (0.2 = 20 %%)")
    parser.add_argument("--min_recall", type=float, default=0.5,
                        help="Recall minimum pour valider le modèle")
    parser.add_argument("--model_name", type=str, default="heart-failure-model",
                        help="Nom du modèle dans le registre")
    parser.add_argument("--register", action="store_true",
                        help="Enregistre le modèle dans le registre s'il est validé")
    return parser.parse_args()


def load_data(path):
    """2. Chargement du CSV : pas de index_col=0, la 1re colonne (age) est une variable."""
    return pd.read_csv(path)


def train_model(X_train, y_train, class_weight):
    """3. Entraînement : standardisation + régression logistique."""
    # "none" (texte reçu en paramètre) -> None (valeur attendue par scikit-learn)
    cw = None if class_weight == "none" else class_weight
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=cw, max_iter=1000)
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """4. Évaluation sur le jeu de test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def main():
    args = parse_args()
    print(f"Paramètres class_weight={args.class_weight}, test_size={args.test_size}, "
          f"min_recall={args.min_recall}, register={args.register}")

    df = load_data(args.data)
    print(f"Données : {df.shape[0]} lignes, {df.shape[1]} colonnes, "
          f"taux de décès = {df[TARGET].mean():.1%}")
    # Variables converties en décimaux : schéma d'entrée plus simple au déploiement
    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Tout ce qui est enregistré dans ce bloc est rattaché au run MLflow
    # (dans un job Azure ML, le run MLflow est le job lui-même)
    with mlflow.start_run():
        # 5. Paramètres du run
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("min_recall", args.min_recall)

        model = train_model(X_train, y_train, args.class_weight)
        metrics = evaluate(model, X_test, y_test)

        for name, value in metrics.items():
            print(name, ":", round(value, 4))

        # 6. Métriques du run (un dictionnaire nom -> valeur)
        mlflow.log_metrics(metrics)

        # 7. Modèle au format MLflow, avec la description de ses entrées et sorties
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        print("Modèle enregistré dans le run MLflow (dossier model/)")

        # 8. Validation : le recall minimum décide du tag et de l'enregistrement
        if metrics["recall"] < args.min_recall:
            mlflow.set_tag("validation", "refusée")
            print(f"Modèle refusé : recall {metrics['recall']:.3f} < {args.min_recall}")
            return

        mlflow.set_tag("validation", "acceptée")
        print(f"Modèle validé : recall {metrics['recall']:.3f} >= {args.min_recall}")
        if args.register:
            run_id = mlflow.active_run().info.run_id
            version = mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
            MlflowClient().set_model_version_tag(
                args.model_name, version.version, "validation", "acceptée"
            )
            print(f"Nouvelle version de {args.model_name} : {version.version}")


if __name__ == "__main__":
    main()
