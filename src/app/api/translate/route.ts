import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import corpus from "@/data/corpus.json";

export const runtime = 'edge';

// 余弦相似度计算函数
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecB[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

const BASE_SYSTEM_PROMPT_RULES = `
你现在是一个“机翻Vlogger语录”转换器。你需要将用户输入的正常中文，转换为一种生硬、滑稽、带有强烈英语母语者直译特征的“机翻中文/翻译腔”。

请严格遵守以下转换规则，模仿给定语料的风格：
1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有“我”、“我的”、“我们”或“他”。
2. 【强行过去式】：在过去的动作后机械地加上“了”（例如：“我开始了看书”、“我做了地铁”、“我感觉了创伤后应激障碍”）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用“然后”、“但是”、“因为”、“所以”串联毫无逻辑关联的琐事流水账。
6. 【抓马情感】：在长文描述的最后或中间，突然插入一句过度严肃、悲观的翻译腔感叹（例如：“我过的是一种充满耻辱的生活”、“我觉得我又存在主义危机”、“我的人生彻底失败了”、“问自己我做这一切是为了什么”）。
7. 【严控长度】：**核心规则！输出的句子数量、段落结构和信息量必须与输入的原句保持绝对一致。如果用户只输入了一句简短的话，你必须且只能输出对应的一句机翻中文，绝不能自行脑补、扩充成一整篇流水账或多出其他句子。**
`;

// 1. 尝试使用主平台：Google Gemini
async function translateWithGemini(apiKey: string, text: string, systemPrompt: string) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ 
      model: "gemini-3.5-flash",
      systemInstruction: systemPrompt,
  });
  const result = await model.generateContent(text);
  const response = await result.response;
  return response.text().trim();
}

// 2. 备用平台：支持 OpenAI 兼容格式的平台（如 DeepSeek, SiliconFlow, OpenAI 等）
async function translateWithFallback(apiKey: string, baseUrl: string, model: string, text: string, systemPrompt: string) {
  const cleanBaseUrl = baseUrl.replace(/\/$/, '');
  const url = `${cleanBaseUrl}/chat/completions`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: model || 'deepseek-chat',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`备用 API 返回错误 (${response.status}): ${errText}`);
  }

  const data = await response.json();
  if (!data.choices || data.choices.length === 0) {
    throw new Error("备用 API 未返回有效的 Choices 数据");
  }
  return data.choices[0].message.content.trim();
}

export async function POST(req: Request) {
  let topExamples: any[] = [];
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ 
        error: "API Key is missing in Cloudflare environment. 请确保已在 Cloudflare 控制台的 Environment Variables 中添加了 GEMINI_API_KEY，且值正确。" 
      }, { status: 500 });
    }

    // 初始化 Gemini API 并计算向量 (注意：向量计算也需要 API 密钥)
    const genAI = new GoogleGenerativeAI(apiKey);
    const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-2" });
    const embedResult = await embeddingModel.embedContent(text);
    const queryVector = embedResult.embedding.values;

    // 计算余弦相似度并排序
    const scoredCorpus = corpus.map(item => {
      const similarity = cosineSimilarity(queryVector, item.embedding);
      return {
        input: item.input,
        output: item.output,
        similarity
      };
    });

    scoredCorpus.sort((a, b) => b.similarity - a.similarity);
    topExamples = scoredCorpus.slice(0, 2);

    // 组装动态 System Prompt
    const systemPrompt = `
${BASE_SYSTEM_PROMPT_RULES}

下面是与你当前翻译文本在语义上最接近的参考语料示例（请重点模仿其措辞和风格）：
${topExamples.map((item, idx) => `
【参考示例 ${idx + 1}（相关度: ${(item.similarity * 100).toFixed(1)}%）】：
正常中文输入：${item.input}
机翻腔翻译输出：${item.output}
`).join("\n")}

不要输出任何解释，不要带有Markdown格式，直接输出转换后的一段话。
`;

    let translatedText = "";
    let providerUsed = "Google Gemini";

    try {
      // 步骤 A: 优先使用 Google Gemini
      translatedText = await translateWithGemini(apiKey, text, systemPrompt);
    } catch (geminiError: any) {
      console.warn("主平台 Gemini 调用失败，尝试启动灾备系统...", geminiError);

      const fallbackKey = process.env.FALLBACK_API_KEY;
      const fallbackUrl = process.env.FALLBACK_BASE_URL || "https://api.deepseek.com/v1";
      const fallbackModel = process.env.FALLBACK_MODEL || "deepseek-chat";

      if (!fallbackKey) {
        // 如果没有配置备用 Key，则直接抛出主平台错误
        throw new Error(`主平台 Gemini 暂时不可用 (${geminiError?.message || "503/429"}), 且未配置备用平台 (FALLBACK_API_KEY)。`);
      }

      // 步骤 B: 启动防线，调用备用平台
      try {
        translatedText = await translateWithFallback(fallbackKey, fallbackUrl, fallbackModel, text, systemPrompt);
        providerUsed = `灾备系统 (${fallbackModel})`;
      } catch (fallbackError: any) {
        throw new Error(`主备双平台均失效。主平台错误: ${geminiError?.message || "503"}; 备用平台错误: ${fallbackError?.message}`);
      }
    }

    return NextResponse.json({ 
      result: translatedText,
      provider: providerUsed,
      reasoning: topExamples.map(item => ({
        input: item.input,
        output: item.output,
        similarity: item.similarity
      }))
    });
  } catch (error: any) {
    console.error("RAG Translation error:", error);
    return NextResponse.json({ 
      error: `翻译失败。错误详情: ${error?.message || error?.toString() || "未知错误"}` 
    }, { status: 500 });
  }
}
