<img src="assets/icon.svg" width="64" height="64" alt="">

# Groupe X - Heart Failure Clinical Records

Membres : Luca Delarue, Alexis Janus, Charlery ... (à compléter)

## Données
299 patients, 12 variables numériques, aucune valeur manquante, aucun doublon. Cible : `DEATH_EVENT` (1 = décès pendant le suivi). Classes : 0 = 203 patients (67,9 %), 1 = 96 patients (32,1 %). Data asset : `heart-failure-data` version 1. Source : UCI, licence CC BY 4.0.

## Métrique choisie et seuil
**Recall, seuil 0,5.** Manquer un patient qui va décéder est plus grave qu'une fausse alerte. Le jeu de test compte 60 patients dont 19 décès : un patient fait varier le recall d'environ 5 points. Sur 50 découpages testés en local, `balanced` reste toujours au-dessus de 0,5 (minimum 0,526), alors que `none` passe 6 fois en dessous. Un seuil plus haut dépendrait surtout du hasard du découpage.

## Résultats
Régression logistique avec `StandardScaler`, `random_state` 42, expérience `heart-default`.

| Run Azure | `class_weight` | `test_size` | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| `mlflow-none-20` | none | 0,2 | 0,817 | 0,786 | 0,579 | 0,667 | 0,859 |
| `mlflow-balanced-20` | balanced | 0,2 | 0,800 | 0,733 | 0,579 | 0,647 | 0,855 |

> À VÉRIFIER avant la remise, puis retirer cet encadré : recall et AUC viennent des runs Azure ; accuracy, precision et F1 viennent du test local sur le même découpage. Recopiez-les depuis l'onglet Metrics. Complétez aussi la version finale : `heart-failure-model` version ..., tag `validation : ...`.

Les deux réglages ont le même recall sur ce découpage (11 décès détectés sur 19). `balanced` est retenu, car son recall moyen est meilleur sur 50 découpages (0,742 contre 0,657).

**Limite :** `time` (durée de suivi) n'est connue qu'à la fin du suivi. Elle est gardée conformément à la fiche, mais elle ne serait pas disponible pour un nouveau patient.

## Preuves
| Étape | Capture | Ce qu'elle montre |
|---|---|---|
| 2 | [Data asset](captures/etape2_data_asset.png) | `heart-failure-data` version 1, type File |
| 3 et 4 | [Runs terminés](captures/etape3_4_runs_completed.png) | `mlflow-none-20` et `mlflow-balanced-20` à l'état Completed |
| 4 | [Comparaison](captures/etape4_comparaison_accuracy.png) | Accuracy des deux runs |
| 5 | [Pipeline](captures/etape5_pipeline.png) | Graphe terminé : préparation, puis entraînement |
| 5 | [Modèle](captures/etape5_modele_v2.png) | `heart-failure-model` version 2, issu de `heart-pipeline` |
| 6 | [Endpoint](captures/etape6_endpoint_url.png) | URL REST de `heart-alexis-11050814` |

## Difficultés rencontrées et comment vous les avez résolues
- **Quota plein (16 vCPU sur 16) :** le job restait en `Queued`, à cause d'endpoints de TP restés actifs. Nous avons supprimé ceux qui ne servaient plus.
- **Connexion à GitHub :** l'autorisation Azure portait un ancien numéro de dépôt. Nous l'avons mise à jour.
- **Mauvais abonnement dans Cloud Shell :** `AuthorizationFailed` sur toutes les commandes. `az account set` a réglé le problème.
- **Faux échec GitHub :** `--stream` renvoyait une erreur alors que le job continuait. Le workflow lit maintenant le statut du job.
- **Noms incohérents :** plusieurs noms de modèle et de data asset coexistaient. Nous avons retenu `heart-failure-model` et `heart-failure-data`.
- **Reprise du TP :** `index_col=0` aurait fait disparaître `age` dans `train.py`. Nous l'avons retiré, et `prep.py` écrit avec `index=False`.

## Lancer l'application
Prérequis : Node.js 20.6 ou plus récent. Copiez `.env.example` vers `.env`, renseignez `AZURE_ML_KEY` et vérifiez `AZURE_ML_ENDPOINT`, puis lancez `npm start` et ouvrez <http://localhost:3001>. Tests : `npm test`.

La clé reste côté serveur. Ne publiez pas le fichier `.env`.
