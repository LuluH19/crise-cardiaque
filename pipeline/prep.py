"""
prep.py - Étape 1 du pipeline : préparation des données
Lit le data asset, supprime les lignes incomplètes et les doublons,
puis écrit un fichier propre que l'étape d'entraînement lira.

Changement par rapport au TP : ce jeu n'a pas de colonne d'identifiant, donc pas
de index_col=0 à la lecture et index=False à l'écriture. Sans cela, l'index
deviendrait une colonne de plus et donc une fausse variable du modèle.
"""
import argparse
import os
import pandas as pd


def main():
    # Paramètres : fichier d'entrée et dossier de sortie (fournis par Azure ML)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Dossier de sortie")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print("Lignes en entrée:", len(df))

    # Suppression des lignes qui contiennent des valeurs manquantes
    df = df.dropna()

    # Suppression des lignes en double
    df = df.drop_duplicates()

    print("Lignes en sortie:", len(df))

    # Écriture du fichier propre dans le dossier de sortie
    os.makedirs(args.output, exist_ok=True)
    chemin_sortie = os.path.join(args.output, "clean.csv")
    df.to_csv(chemin_sortie, index=False)
    print("Fichier écrit:", chemin_sortie)


if __name__ == "__main__":
    main()
