import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createServer } from 'node:http';
import { createServer as createAppServer } from './server.js';

let upstream;
let app;
let appUrl;
let receivedAuthorization;
let receivedPayload;

before(async () => {
  upstream = createServer(async (request, response) => {
    receivedAuthorization = request.headers.authorization;
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    receivedPayload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ prediction: 1, probability: 0.8 }));
  });
  await new Promise(resolve => upstream.listen(0, '127.0.0.1', resolve));

  const upstreamUrl = `http://127.0.0.1:${upstream.address().port}/score`;
  app = createAppServer({ endpoint: upstreamUrl, key: 'test-secret' });
  await new Promise(resolve => app.listen(0, '127.0.0.1', resolve));
  appUrl = `http://127.0.0.1:${app.address().port}`;
});

after(async () => {
  await Promise.all([upstream, app].map(server => new Promise((resolve, reject) => {
    server.close(error => error ? reject(error) : resolve());
  })));
});

test('transmet la prédiction et la clé côté serveur à Azure ML', async () => {
  const payload = { age: 60, diabetes: 0 };
  const response = await fetch(`${appUrl}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { prediction: 1, probability: 0.8 });
  assert.equal(receivedAuthorization, 'Bearer test-secret');
  assert.deepEqual(receivedPayload, { data: [payload] });
});

test('rejette un corps JSON invalide avant l’appel à Azure ML', async () => {
  const response = await fetch(`${appUrl}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{'
  });

  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), { error: 'Le JSON envoyé est invalide.' });
});
