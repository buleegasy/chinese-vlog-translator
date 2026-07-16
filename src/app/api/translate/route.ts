import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import embeddedCorpus from "@/data/corpus.json"; // 包含了 Cloudflare BGE-M3 向量的数据

export const runtime = 'edge';

// ==================== 余弦相似度 (Cloudflare AI 向量核心) ====================
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 0;
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 实时获取用户的输入向量
async function getCloudflareEmbedding(text: string, accountId: string, apiToken: string) {
  const url = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/@cf/baai/bge-m3`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: [text] })
  });
  if (!response.ok) throw new Error(`CF Embedding Failed: ${response.status}`);
  const result = await response.json();
  if (!result.success) throw new Error(`CF Embedding Error: ${JSON.stringify(result.errors)}`);
  return result.result.data[0];
}

// 超时控制包装器
function timeoutPromise<T>(promise: Promise<T>, ms: number, errorMsg: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(errorMsg));
    }, ms);

    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

const BASE_SYSTEM_PROMPT_RULES = `
你现在是一个"机翻Vlogger语录"转换器。你需要将用户输入的正常中文，转换为一种生硬、滑稽、带有强烈英语母语者直译特征的"机翻中文/翻译腔"。

## ⚡ 最高优先级规则（必须严格遵守，高于以下所有规则）

A. 【参考示例优先】：我会提供若干"参考示例"，这些示例展示了目标风格。你必须**直接借用**这些示例中出现的特殊词汇、经典表达和句式结构。例如，若示例中出现"我在中国没有工作的天"，而用户输入也是关于"没有工作"的，你必须照搬"我在中国没有工作的天"这个句式，而不是自己另造一句。**把示例当作必须套用的模板，而不仅仅是参考。**

B. 【信息完整保留】：输入中每一个细节（人物、动作、地点、时间、情绪）都必须出现在输出中。**绝对不允许省略或合并原文中的任何信息。**输入有几个动作，输出就必须有几个对应的动作，一个都不能少。

## 通用翻译腔规则

1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有"我"、"我的"、"我们"或"他"。
2. 【强行过去式】：在过去的动作后机械地加上"了"（例如："我开始了看书"、"我做了地铁"、"我感觉了创伤后应激障碍"）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。若参考示例中已经对某个词进行了降维，必须直接使用示例中的降维结果。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用"然后"、"但是"、"因为"、"所以"串联毫无逻辑关联的琐事。
6. 【抓马情感】：**仅在用户原句带有明显负面情绪或属于长篇流水账时**，才可插入一句过度严肃的翻译腔感叹（例如："我过的是一种充满耻辱的生活"、"我觉得我又存在主义危机"）。**普通中性的日常句子严禁捏造这些词汇。**
7. 【严控长度】：输出的句子数量必须与输入保持绝对一致。输入一句，输出一句；输入三句，输出三句，不多也不少。
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

// 2. 备用平台/主力中转：支持 OpenAI 兼容格式的平台（如 OpenRouter, DeepSeek 等）
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
      model: model || 'minimax/minimax-m2.5',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`备用/主力中转 API 返回错误 (${response.status}): ${errText}`);
  }

  const data = await response.json();
  if (!data.choices || data.choices.length === 0) {
    throw new Error("API 未返回有效的 Choices 数据");
  }
  return data.choices[0].message.content.trim();
}

export async function POST(req: Request) {
  let topExamples: any[] = [];
  let providerUsed = "";
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 });
    }

    console.log(`\n🕒 [${new Date().toISOString()}] === 开始 RAG 翻译推演 ===`);
    console.log(`[1] 收到用户输入: "${text}"`);

    // ==================== 强制要求 智能 RAG 向量检索 ====================
    // 若网络或认证失败，直接抛出异常，不再降级！
    const cfAccountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const cfApiToken = process.env.CLOUDFLARE_API_TOKEN;
    
    if (!cfAccountId || !cfApiToken) {
      throw new Error("Cloudflare RAG 失败：未配置 Account ID 或 API Token，无法进行高精度检索。");
    }

    console.log(`[2] 开始请求 Cloudflare API 获取输入向量...`);
    const t0 = Date.now();
    const queryVector = await getCloudflareEmbedding(text, cfAccountId, cfApiToken);
    console.log(`[3] 获取向量成功，耗时: ${Date.now() - t0}ms, 向量维度: ${queryVector.length}`);

    const corpusArray = Array.isArray(embeddedCorpus) ? embeddedCorpus : (embeddedCorpus as any).default;

    if (!corpusArray || corpusArray.length === 0 || !corpusArray[0].embedding) {
      throw new Error("Cloudflare RAG 失败：本地语料库 corpus.json 未包含向量数据或加载失败！");
    }

    console.log(`[4] 开始在本地进行余弦相似度匹配（语料库大小：${corpusArray.length} 条）...`);
    const t1 = Date.now();
    
    providerUsed += "(RAG: Cloudflare Vectors) ";
    const scoredCorpus = corpusArray.map((item: any) => {
      const similarity = cosineSimilarity(queryVector, item.embedding);
      return {
        input: item.input,
        output: item.output,
        similarity
      };
    });
    
    scoredCorpus.sort((a: any, b: any) => b.similarity - a.similarity);
    topExamples = scoredCorpus.slice(0, 2);
    
    console.log(`[5] 匹配完成，耗时: ${Date.now() - t1}ms。`);
    console.log(`最高相似度: ${(topExamples[0].similarity * 100).toFixed(2)}% | 命中: ${topExamples[0].input}`);

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

    const useFallbackAsPrimary = process.env.USE_FALLBACK_AS_PRIMARY === "true" || !process.env.GEMINI_API_KEY;
    const fallbackKey = process.env.FALLBACK_API_KEY;
    const fallbackUrl = process.env.FALLBACK_BASE_URL || "https://openrouter.ai/api/v1";
    // 默认模型设为用户指定的 MiniMax-M2.5
    const fallbackModel = process.env.FALLBACK_MODEL || "google/gemini-2.5-flash";

    // 2. 路由分流与翻译执行
    console.log(`[6] 开始请求大语言模型进行翻译...`);
    const t2 = Date.now();
    
    if (useFallbackAsPrimary && fallbackKey) {
      // 模式 A：直接走 OpenRouter (使用 MiniMax / 其他模型)
      translatedText = await translateWithFallback(fallbackKey, fallbackUrl, fallbackModel, text, systemPrompt);
      const latency = Date.now() - t2;
      providerUsed += `| Model: OpenRouter (${fallbackModel}, ${latency}ms)`;
      console.log(`[7] 翻译完成 (OpenRouter - ${fallbackModel})，耗时: ${latency}ms`);
    } else {
      // 模式 B：优先 Gemini，失败后降级 OpenRouter
      const apiKey = process.env.GEMINI_API_KEY || "";
      try {
        translatedText = await timeoutPromise(
          translateWithGemini(apiKey, text, systemPrompt),
          8000,
          "Gemini 响应超时 (8s)"
        );
        const latency = Date.now() - t2;
        providerUsed += `| Model: Google Gemini (${latency}ms)`;
        console.log(`[7] 翻译完成 (Google Gemini)，耗时: ${latency}ms`);
      } catch (geminiError: any) {
        const geminiLatency = Date.now() - t2;
        console.warn(`主平台 Gemini 调用失败 (耗时: ${geminiLatency}ms)，自动切换至中转备用平台...`, geminiError.message || geminiError);

        if (!fallbackKey) {
          throw new Error(`主平台 Gemini 暂时不可用 (${geminiError?.message || "Timeout"}), 且未配置备用平台 (FALLBACK_API_KEY)。`);
        }

        try {
          console.log(`[6.5] 开始请求灾备系统 (${fallbackModel})...`);
          const t3 = Date.now();
          translatedText = await translateWithFallback(fallbackKey, fallbackUrl, fallbackModel, text, systemPrompt);
          const fallbackLatency = Date.now() - t3;
          providerUsed += `| Model: 灾备系统 (${fallbackModel}, ${fallbackLatency}ms)`;
          console.log(`[7] 翻译完成 (灾备系统 - ${fallbackModel})，耗时: ${fallbackLatency}ms`);
        } catch (fallbackError: any) {
          throw new Error(`主备双平台均失效。主平台错误: ${geminiError?.message || "503/Timeout"}; 备用平台错误: ${fallbackError?.message}`);
        }
      }
    }

    const totalTime = Date.now() - t0;
    console.log(`[8] 总流程结束，RAG+翻译总耗时: ${totalTime}ms\n`);

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
