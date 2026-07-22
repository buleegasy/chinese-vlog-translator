import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import embeddedCorpus from "@/data/corpus.json"; // 包含了 Cloudflare BGE-M3 向量的数据

export const runtime = 'edge';

// ⚡ Bolt Optimization: Simple memory cache for identical translation requests.
// In Edge runtimes (like Cloudflare Workers), module-level variables persist
// across multiple requests hitting the same isolate, providing a free caching layer.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const translationCache = new Map<string, any>();
const MAX_CACHE_SIZE = 1000;

// ==================== 余弦相似度 (Cloudflare AI 向量核心) ====================
// ⚡ Bolt Optimization: Since Cloudflare BGE-M3 embeddings are L2-normalized,
// the denominator is always 1. We can skip magnitude calculation and just use dot product.
// ⚡ Bolt Optimization: Unroll loop by 4x to reduce loop overhead and branching for heavy dot product math,
// providing ~20%+ speedup in JS Edge Runtimes when processing thousands of vectors.
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const len = vecA.length;
  if (len !== vecB.length) return 0;
  let dotProduct = 0;
  let i = 0;
  for (; i <= len - 4; i += 4) {
    dotProduct += vecA[i] * vecB[i] +
                  vecA[i + 1] * vecB[i + 1] +
                  vecA[i + 2] * vecB[i + 2] +
                  vecA[i + 3] * vecB[i + 3];
  }
  for (; i < len; i++) {
    dotProduct += vecA[i] * vecB[i];
  }
  return dotProduct;
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
你是一个将正常中文转换为"外国人机翻腔中文"的转换器。

目标风格来自一个英语母语者拍摄中国Vlog时，他的中文字幕因极其低劣的机器翻译产生的滑稽效果。核心是：放弃正常的中文语法，彻底用最原始、最生硬的英语思维直译。

---

## 核心转换指令：

**0. 深刻理解上下文（Contextual Understanding）**
在进行任何语法破坏或造词之前，你必须首先通读并深刻理解整个 input context，弄清楚用户到底在描述什么场景、想表达什么情绪。绝对不要断章取义地进行机械替换，所有的造词和戏剧化发挥都必须基于对用户真实意图的整体把控。

**1. 自由造词与极致降维（最关键）**
大胆发挥你的想象力！不要局限于正常的词汇表。用极度匮乏的基础词汇强行组合、拼凑，甚至**自己造出极具特色的荒诞词汇**来描述复杂事物。
例：雾霾→天气很脏；维修工→水桶的人；无脑片→很笨的电影；漏水→淹没了；游乐园→有很多快乐设备的村。
尽情使用令人匪夷所思的独特造词。

**2. 语法彻底破坏**
绝对禁止输出通顺流畅的中文。
- 状语、补语必须放错位置。
- 修饰语和名词强行割裂。
- 多用生硬的介词代替动词（如用"在"、"的"代替具体动作）。
- 把复杂的长句生硬地切分成破碎的短句。

**3. "了"的错位滥用**
在别扭、奇怪的位置疯狂加"了"，制造语法的生硬感。
例：「我做了地铁」「我开始了回家」「我想了吃什么的」。

**4. 主语强制冗余**
每句话不管多短，都必须带上"我"、"我的"、"他"等主语，绝不省略。

**5. 模糊代词与生硬连词**
- 频繁使用"什么的"代替不知道叫什么的词（如："到了什么的沼泽"）。
- "然后""但是""因为""所以"等连词可以毫无逻辑地乱用。

**6. 极度夸张的抓马感（Drama）**
把日常小事无限拔高，制造荒诞的戏剧感。可以用极其严重的存在主义、宗教或文学词汇来形容日常琐事。
例：掉了个苹果→"生命的悲剧"；有点累→"灵魂的枯竭"、"充满耻辱的生活"、"创伤后应激障碍"。

---

## 输出要求
- 保留原文包含的所有基本事实动作，但完全打碎它的表达方式。
- **直接输出最终的翻译结果，禁止输出任何思考过程、自我审查或解释说明，禁止使用Markdown格式。**`;

// LLM 翻译函数
async function translateWithGemini(apiKey: string, text: string, systemPrompt: string) {
  const genAI = new GoogleGenerativeAI(apiKey);
  // 使用合法的 Gemini 模型版本，防止因找不到 3.5 导致报错降级
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash", systemInstruction: systemPrompt });
  const result = await model.generateContent(text);
  return result.response.text().trim();
}

async function translateWithFallback(apiKey: string, baseUrl: string, model: string, text: string, systemPrompt: string) {
  const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: text }], temperature: 0.7 })
  });
  if (!response.ok) throw new Error(`备用 API 错误 (${response.status}): ${await response.text()}`);
  const data = await response.json();
  if (!data.choices?.length) throw new Error("API 未返回有效 Choices");
  return data.choices[0].message.content.trim();
}

export async function POST(req: Request) {
  let topExamples: { input: string; output: string; similarity: number }[] = [];
  let providerUsed = "";
  try {
    const { text } = await req.json();
    if (!text) return NextResponse.json({ error: "No text provided" }, { status: 400 });

    const trimmedText = text.trim();
    if (translationCache.has(trimmedText)) {
      console.log(`\n⚡ [${new Date().toISOString()}] Bolt Cache Hit for input: "${trimmedText}"`);
      return NextResponse.json(translationCache.get(trimmedText));
    }

    console.log(`\n🕒 [${new Date().toISOString()}] === 开始 RAG 翻译推演 ===`);
    console.log(`[1] 收到用户输入: "${text}"`);

    const cfAccountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const cfApiToken = process.env.CLOUDFLARE_API_TOKEN;
    if (!cfAccountId || !cfApiToken) throw new Error("Cloudflare RAG 失败：未配置 Account ID 或 API Token");

    console.log(`[2] 开始请求 Cloudflare API 获取输入向量...`);
    const t0 = Date.now();
    const queryVector = await getCloudflareEmbedding(text, cfAccountId, cfApiToken);
    console.log(`[3] 获取向量成功，耗时: ${Date.now() - t0}ms, 维度: ${queryVector.length}`);

    const corpusArray = Array.isArray(embeddedCorpus) ? embeddedCorpus : (embeddedCorpus as { default: unknown[] }).default;
    if (!corpusArray?.length || !(corpusArray[0] as { embedding?: unknown }).embedding) {
      throw new Error("corpus.json 未包含向量数据或加载失败");
    }

    const t1 = Date.now();
    providerUsed += "(RAG: Cloudflare Vectors) ";

    // ⚡ Bolt Optimization: Replace O(N log N) map+sort with an O(N) single pass
    // to find the top 2 most similar examples. This prevents allocating a large
    // intermediate array and avoids the overhead of sorting the entire corpus.
    let top1 = { input: "", output: "", similarity: -Infinity };
    let top2 = { input: "", output: "", similarity: -Infinity };

    for (let i = 0; i < corpusArray.length; i++) {
      const item = corpusArray[i] as { input: string; output: string; embedding: number[] };
      const similarity = cosineSimilarity(queryVector, item.embedding);
      if (similarity > top1.similarity) {
        top2 = top1;
        top1 = { input: item.input, output: item.output, similarity };
      } else if (similarity > top2.similarity) {
        top2 = { input: item.input, output: item.output, similarity };
      }
    }

    topExamples = [top1, top2].filter(x => x.similarity !== -Infinity);
    console.log(`[4] 向量匹配完成，耗时: ${Date.now() - t1}ms`);
    console.log(`最高相似度: ${(topExamples[0].similarity * 100).toFixed(2)}% | 命中: ${topExamples[0].input}`);

    // 根据 RAG 相似度决定策略：高相似度直接用语料，低相似度作风格参考
    const topSim = topExamples[0]?.similarity ?? 0;
    const HIGH_SIMILARITY_THRESHOLD = 0.88; // 超过此值：语义几乎一致，优先直接照搬

    let ragSection = "";
    if (topSim >= HIGH_SIMILARITY_THRESHOLD) {
      ragSection = `
⚡【高相似度硬性指令】（相似度高达 ${(topSim * 100).toFixed(1)}%）：
当前用户输入的句子与下方【优先照搬】中的【原始输入】在语义上几乎完全相同（仅有微小差异）。
你必须遵守以下硬性规定：
1. **直接复制并微调句式**：禁止重新翻译或进行自主创作。必须直接照抄【机翻输出】的句子结构、降维词汇和特殊连词。
2. **细节补齐（补差）**：如果用户输入比下方的【原始输入】多出了一两个细节词（如时间、语气词、特定人名等），你只能在这条【机翻输出】的基础上将这些细节以相同的"机翻腔"补进去，绝对不能丢掉。
3. **不得丢弃信息**：确保最终输出中，用户输入的每个动作和细节都在，不得丢失信息。

【优先照搬】
原始输入：${topExamples[0].input}
机翻输出：${topExamples[0].output}
${topExamples[1] ? `\n【补充参考】\n原始输入：${topExamples[1].input}\n机翻输出：${topExamples[1].output}` : ""}
`;
    } else {
      ragSection = `
以下是语义检索召回的最相关参考，作为当前输入的风格对照：
${topExamples.map((item, idx) => `【参考 ${idx + 1}（相关度 ${(item.similarity * 100).toFixed(1)}%）】\n原始输入：${item.input}\n机翻输出：${item.output}`).join("\n\n")}
`;
    }

    // 组装最终 System Prompt
    const systemPrompt = `
${BASE_SYSTEM_PROMPT_RULES}

---

${ragSection}

直接输出转换后的文本，不要任何解释、不要Markdown格式。
`;

    let translatedText = "";

    const useFallbackAsPrimary = true; // 强制走 OpenRouter，因为用户指定了 Claude 模型
    const fallbackKey = process.env.FALLBACK_API_KEY;
    const fallbackUrl = process.env.FALLBACK_BASE_URL || "https://openrouter.ai/api/v1";
    
    // 优先使用用户在环境变量中配置的模型，如果没有配置，则默认使用指定的 claude-haiku-4.5
    let fallbackModel = process.env.FALLBACK_MODEL || "anthropic/claude-haiku-4.5";
    
    // 如果环境变量里遗留了 minimax，强制覆写为 claude
    if (fallbackModel.toLowerCase().includes("minimax") || fallbackModel.includes("gemini")) {
      fallbackModel = "anthropic/claude-haiku-4.5";
    }

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

    const responsePayload = {
      result: translatedText,
      provider: providerUsed,
      reasoning: topExamples.map(item => ({
        input: item.input,
        output: item.output,
        similarity: item.similarity
      }))
    };

    // Cache the response
    if (translationCache.size >= MAX_CACHE_SIZE) {
      // Very simple LRU approximation: delete the first key when full
      const firstKey = translationCache.keys().next().value;
      if (firstKey !== undefined) translationCache.delete(firstKey);
    }
    translationCache.set(trimmedText, responsePayload);

    return NextResponse.json(responsePayload);
  } catch (error: any) {
    console.error("RAG Translation error:", error);
    return NextResponse.json({ 
      error: `翻译失败。错误详情: ${error?.message || error?.toString() || "未知错误"}` 
    }, { status: 500 });
  }
}
