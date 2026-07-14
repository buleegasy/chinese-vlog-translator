import fs from 'fs/promises';

const ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
const API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;

async function main() {
  const rawData = JSON.parse(await fs.readFile('src/data/raw_corpus.json', 'utf8'));
  const inputs = rawData.map(item => item.input);
  
  console.log(`Generating embeddings for ${inputs.length} sentences using Cloudflare @cf/baai/bge-m3...`);
  
  const response = await fetch(`https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/baai/bge-m3`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: inputs })
  });

  if (!response.ok) {
    console.error('Failed to fetch:', response.statusText);
    console.error(await response.text());
    return;
  }

  const result = await response.json();
  if (!result.success) {
    console.error('API Error:', result.errors);
    return;
  }

  const embeddings = result.result.data; // Array of arrays

  const corpusWithEmbeddings = rawData.map((item, index) => ({
    ...item,
    embedding: embeddings[index]
  }));

  await fs.writeFile('src/data/corpus.json', JSON.stringify(corpusWithEmbeddings, null, 2));
  console.log(`Successfully wrote ${corpusWithEmbeddings.length} items with embeddings to src/data/corpus.json`);
}

main();
