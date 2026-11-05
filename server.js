import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(fileURLToPath(import.meta.url));
const maxRequestSize = 64 * 1024;

function sendJson(response, status, data) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(data));
}

async function readJsonRequest(request) {
  if (!request.headers['content-type']?.startsWith('application/json')) {
    const error = new Error('Le contenu doit être au format JSON.');
    error.status = 415;
    throw error;
  }

  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxRequestSize) {
      const error = new Error('La requête est trop volumineuse.');
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }

  let data;
  try {
    data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    const error = new Error('Le JSON envoyé est invalide.');
    error.status = 400;
    throw error;
  }
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    const error = new Error('Le JSON doit être un objet.');
    error.status = 400;
    throw error;
  }
  return data;
}

export function createServer({ endpoint, key }) {
  return createHttpServer(async (request, response) => {
    if (request.method === 'GET' && request.url === '/') {
      try {
        const html = await readFile(join(root, 'index.html'));
        response.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'X-Content-Type-Options': 'nosniff'
        });
        response.end(html);
      } catch (error) {
        console.error('Impossible de lire index.html :', error);
        sendJson(response, 500, { error: 'Impossible de charger la page.' });
      }
      return;
    }

    // Seuls les fichiers PNG placés directement dans captures/ sont servis.
    const capture = request.url.match(/^\/captures\/([a-z0-9_]+\.png)$/);
    if (request.method === 'GET' && capture) {
      try {
        const image = await readFile(join(root, 'captures', capture[1]));
        response.writeHead(200, {
          'Content-Type': 'image/png',
          'X-Content-Type-Options': 'nosniff'
        });
        response.end(image);
      } catch {
        sendJson(response, 404, { error: 'Ressource introuvable.' });
      }
      return;
    }

    if (request.url !== '/api/predict') {
      sendJson(response, 404, { error: 'Ressource introuvable.' });
      return;
    }
    if (request.method !== 'POST') {
      response.setHeader('Allow', 'POST');
      sendJson(response, 405, { error: 'Méthode non autorisée.' });
      return;
    }

    try {
      const body = await readJsonRequest(request);
      const upstream = await fetch(endpoint, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: [body] })
      });
      const responseBody = Buffer.from(await upstream.arrayBuffer());
      response.writeHead(upstream.status, {
        'Content-Type': upstream.headers.get('content-type') || 'application/json',
        'X-Content-Type-Options': 'nosniff'
      });
      response.end(responseBody);
    } catch (error) {
      if (error.status) {
        sendJson(response, error.status, { error: error.message });
        return;
      }
      console.error('Échec de l’appel à Azure ML :', error);
      sendJson(response, 502, { error: 'Impossible de joindre le service Azure ML.' });
    }
  });
}

function start() {
  const endpointValue = process.env.AZURE_ML_ENDPOINT;
  const key = process.env.AZURE_ML_KEY;
  if (!endpointValue || !key) {
    console.error('Définissez AZURE_ML_ENDPOINT et AZURE_ML_KEY dans le fichier .env.');
    process.exitCode = 1;
    return;
  }

  let endpoint;
  try {
    endpoint = new URL(endpointValue);
  } catch {
    console.error('AZURE_ML_ENDPOINT doit être une URL valide.');
    process.exitCode = 1;
    return;
  }
  if (endpoint.protocol !== 'https:') {
    console.error('AZURE_ML_ENDPOINT doit utiliser HTTPS.');
    process.exitCode = 1;
    return;
  }

  const port = Number(process.env.PORT || 3001);
  const server = createServer({ endpoint: endpoint.href, key });
  server.on('error', error => {
    console.error('Impossible de démarrer le serveur :', error);
    process.exitCode = 1;
  });
  server.listen(port, process.env.HOST || '127.0.0.1', () => {
    console.log(`Application disponible sur http://${process.env.HOST || '127.0.0.1'}:${port}`);
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  start();
}
