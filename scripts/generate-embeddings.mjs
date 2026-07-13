import fs from 'fs';
import path from 'path';
import { GoogleGenerativeAI } from "@google/generative-ai";

// 1. 读取环境变量中的 API Key
let apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  try {
    const envPath = path.resolve('.env.local');
    if (fs.existsSync(envPath)) {
      const envContent = fs.readFileSync(envPath, 'utf8');
      const match = envContent.match(/GEMINI_API_KEY\s*=\s*(.*)/);
      if (match && match[1]) {
        apiKey = match[1].trim();
      }
    }
  } catch (err) {
    console.error("无法读取 .env.local 文件:", err);
  }
}

if (!apiKey) {
  console.error("错误: 未找到 GEMINI_API_KEY 环境变量或 .env.local 配置。");
  process.exit(1);
}

// 2. 初始化 Gemini API
const genAI = new GoogleGenerativeAI(apiKey);
// 使用 2026 推荐的 gemini-embedding-2 模型
const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-2" });

async function main() {
  const rawCorpusPath = path.resolve('src/data/raw_corpus.json');
  const outCorpusPath = path.resolve('src/data/corpus.json');

  if (!fs.existsSync(rawCorpusPath)) {
    console.error(`错误: 找不到原始语料文件: ${rawCorpusPath}`);
    process.exit(1);
  }

  const rawCorpus = JSON.parse(fs.readFileSync(rawCorpusPath, 'utf8'));
  console.log(`开始为 ${rawCorpus.length} 条语料生成向量...`);

  const corpusWithEmbeddings = [];

  for (let i = 0; i < rawCorpus.length; i++) {
    const item = rawCorpus[i];
    console.log(`[${i + 1}/${rawCorpus.length}] 正在计算向量: "${item.input.substring(0, 15)}..."`);
    
    try {
      // 调用 Gemini 生成 Embedding
      const result = await embeddingModel.embedContent(item.input);
      const embedding = result.embedding.values;
      
      corpusWithEmbeddings.push({
        id: i + 1,
        input: item.input,
        output: item.output,
        embedding: embedding
      });
    } catch (error) {
      console.error(`计算第 ${i + 1} 条语料向量失败:`, error);
      process.exit(1);
    }
  }

  // 3. 写入带有向量的最终语料库
  fs.writeFileSync(outCorpusPath, JSON.stringify(corpusWithEmbeddings, null, 2), 'utf8');
  console.log(`成功！已将带向量的语料库写入: ${outCorpusPath}`);
}

main().catch(console.error);
