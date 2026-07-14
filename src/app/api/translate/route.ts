import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import rawCorpus from "@/data/raw_corpus.json";

export const runtime = 'edge';

// 本地 Bi-Gram 分词与余弦/Jaccard 相似度计算（零延迟、零API成本、100%可靠）
function getBiGrams(text: string): string[] {
  // 只保留中文、英文和数字
  const clean = text.replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, '');
  const grams: string[] = [];
  for (let i = 0; i < clean.length - 1; i++) {
    grams.push(clean.substring(i, i + 2));
  }
  // 如果字数太少（比如只有一个字或为空），退化为单字分词
  if (grams.length === 0 && clean.length > 0) {
    return clean.split('');
  }
  return grams;
}

function calculateJaccardSimilarity(query: string, target: string): number {
  const qGrams = new Set(getBiGrams(query));
  const tGrams = new Set(getBiGrams(target));
  
  if (qGrams.size === 0 || tGrams.size === 0) return 0;
  
  const intersection = new Set([...qGrams].filter(x => tGrams.has(x)));
  const union = new Set([...qGrams, ...tGrams]);
  
  return intersection.size / union.size;
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
你现在是一个“机翻Vlogger语录”转换器。你需要将用户输入的正常中文，转换为一种生硬、滑稽、带有强烈英语母语者直译特征的“机翻中文/翻译腔”。

请严格遵守以下转换规则，模仿给定语料的风格：
1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有“我”、“我的”、“我们”或“他”。
2. 【强行过去式】：在过去的动作后机械地加上“了”（例如：“我开始了看书”、“我做了地铁”、“我感觉了创伤后应激障碍”）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用“然后”、“但是”、“因为”、“所以”串联毫无逻辑关联的琐事流水账。
6. 【抓马情感】：**仅在用户原句带有明显的负面情绪（如劳累、伤心、倒霉、迷茫）或属于长篇流水账日记时**，才可在适当位置插入一句过度严肃、悲观的翻译腔感叹（例如：“我过的是一种充满耻辱的生活”、“我觉得我又存在主义危机”、“我的人生彻底失败了”）。**如果原句只是中性或积极的普通日常（如：吃个饭、出门散步），必须保持客观平淡，绝对不要强行捏造并插入这些存在主义悲剧词汇！**
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
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 });
    }

    // 1. 本地 NLP 语义检索（Bi-gram Jaccard）
    // 零网络开销，完美防范限流，秒速召回与输入最匹配的单句教科书
    const scoredCorpus = rawCorpus.map(item => {
      const similarity = calculateJaccardSimilarity(text, item.input);
      return {
        input: item.input,
        output: item.output,
        similarity
      };
    });

    // 按相似度降序排序
    scoredCorpus.sort((a, b) => b.similarity - a.similarity);
    
    // 如果没有匹配到任何有交集的词，默认取前 2 条，否则取匹配度最高的前 2 条
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
    let providerUsed = "";

    const useFallbackAsPrimary = process.env.USE_FALLBACK_AS_PRIMARY === "true" || !process.env.GEMINI_API_KEY;
    const fallbackKey = process.env.FALLBACK_API_KEY;
    const fallbackUrl = process.env.FALLBACK_BASE_URL || "https://openrouter.ai/api/v1";
    // 默认模型设为用户指定的 MiniMax-M2.5
    const fallbackModel = process.env.FALLBACK_MODEL || "minimax/minimax-m2.5";

    // 2. 路由分流与翻译执行
    if (useFallbackAsPrimary && fallbackKey) {
      // 模式 A：直接走 OpenRouter (使用 MiniMax / 其他模型)
      translatedText = await translateWithFallback(fallbackKey, fallbackUrl, fallbackModel, text, systemPrompt);
      providerUsed = `OpenRouter (${fallbackModel})`;
    } else {
      // 模式 B：优先 Gemini，失败后降级 OpenRouter
      const apiKey = process.env.GEMINI_API_KEY || "";
      try {
        translatedText = await timeoutPromise(
          translateWithGemini(apiKey, text, systemPrompt),
          2500,
          "Gemini 响应超时 (2.5s)"
        );
        providerUsed = "Google Gemini";
      } catch (geminiError: any) {
        console.warn("主平台 Gemini 调用失败，自动切换至中转备用平台...", geminiError);

        if (!fallbackKey) {
          throw new Error(`主平台 Gemini 暂时不可用 (${geminiError?.message || "Timeout"}), 且未配置备用平台 (FALLBACK_API_KEY)。`);
        }

        try {
          translatedText = await translateWithFallback(fallbackKey, fallbackUrl, fallbackModel, text, systemPrompt);
          providerUsed = `灾备系统 (${fallbackModel})`;
        } catch (fallbackError: any) {
          throw new Error(`主备双平台均失效。主平台错误: ${geminiError?.message || "503/Timeout"}; 备用平台错误: ${fallbackError?.message}`);
        }
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
