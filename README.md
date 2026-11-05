<img src="assets/icon.svg" width="64" height="64" alt="">

# crise-cardiaque

## Lancer l’application

Prérequis : Node.js 20.6 ou plus récent.

1. Copiez `.env.example` vers `.env`.
2. Dans `.env`, renseignez `AZURE_ML_KEY` avec la clé de l’endpoint Azure ML. Vérifiez aussi `AZURE_ML_ENDPOINT`.
3. Lancez `npm start`, puis ouvrez <http://localhost:3001>.

Le serveur envoie les données au format Azure ML `{ "data": [patient] }` et ajoute la clé à l’en-tête `Authorization: Bearer`. La clé reste côté serveur et n’est pas envoyée au navigateur. Ne publiez pas le fichier `.env`.
Pour lancer les tests, exécutez `npm test`.

<sub>visuel généré avec l'aide de l'intéligence artificiel</sub>
