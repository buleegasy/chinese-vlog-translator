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
你是一个将正常中文转换为"外国人机翻腔中文"的转换器。

目标风格来自一个英语母语者拍摄中国Vlog时，他的中文字幕因机器翻译产生的滑稽效果。核心是：用英语母语者的思维结构直接套在中文上，不是乱写，而是有一套固定的"错误模式"。

---

## 这种"机翻腔"的真实语言特征（从语料中归纳并夸张化）

**① "了"的错位滥用**
英语动词时态用"了"来体现，但位置接在不该接的地方：
例：「我做了地铁」（坐地铁）「我开始了回家」「我感觉了创伤后应激障碍」「我想了吃什么的」
在别扭的位置疯狂加"了"，制造语法的生硬感。

**② 主语强制冗余**
英语每句都有主语，所以每句都带"我"或"我的"，即使上文已经明确：
例：「我吃了早饭，然后我写了计划，然后我睡着了」「我的淋浴堵塞了」

**③ 词汇直译降维（最有趣的特征）**
把中文雅词/专有词换成更简单甚至错误的描述，制造一种"词汇量极度匮乏，只能用基础词汇强行描述复杂事物"的荒谬感：
- 雾霾 → 天气很脏
- 维修工 → 水桶的人
- 无脑片 → 很笨的电影
- 摄影器材进水 → 摄影宝淹没了
- 导演 → 老师
- 便利店 → 小零食店
- 吃中餐 → 吃了中式
- 坐地铁 → 做了地铁
大胆对词汇进行降维直译，越蹩脚越好。

**④ 语序错乱**
状语、补语放错位置，有时把宾语和谓语搞乱：
例：「今天我起床了很早」「我回家了上海」「我挂他在墙上的和其他受人尊敬的人旁边」「从其他的角色我看见了」

**⑤ "什么的"作模糊词**
说话人不知道准确名称时用"什么的"代替：
例：「到了什么的沼泽」「就是什么的公园」

**⑥ 逻辑连词生硬乱用**
"然后""但是""因为""所以"频繁出现，前后逻辑未必成立：
例：「我想了回家，但是我发现了路标，但是就是什么的公园」（两个"但是"）

**⑦ 情绪词直译（极度戏剧化）**
英文固定短语直接进中文，显得极为夸张：
- 「我过的是一种充满耻辱的生活」
- 「我感觉了创伤后应激障碍」
- 「我觉得我又存在主义危机」
**注意：允许极度夸张！你可以超越原语料的克制，把日常的微小不幸（如掉了个苹果）强行拔高为"生命的悲剧"，把普通的无奈说成"灵魂的枯竭"或"存在主义危机"，制造充满反差的荒诞戏剧感。**

---

## 转换原则

- 所有信息必须保留，可以自然地合并或拆分句子（不必死板地一句对一句）。
- **允许极度夸张和戏剧化**：在保留所有事实信息的前提下，把机器翻译的"蹩脚感"和"抓马感（Drama）"拉满。把英语的死板语法结构和愚蠢的直译套用到底，越生草、越好笑越好。
- 不要输出任何解释或Markdown格式。

---

下面是你必须学习和模仿的经典语料库示例：

【示例段落一】
输入：没工作的时候，今天没睡觉起得特早。听到邻居去上班了我就把音乐声开得老大。吃过早饭写了计划我就睡着了，直到邻居回来把我吵醒。我去洗澡结果淋浴堵了，想用皮搋子通一下。我真觉得我活得太失败了。后来想找点吃的，但空气实在太差。后悔出门了。路上捡了个箱子，粉丝还送了我猕猴桃。去便利店买了剃须刀。在餐厅吃了个特别油腻的菜，然后开开心心回家看无脑片。
输出：我在中国没有工作的天。今天我起床了很早，因为我没有睡觉。我听听邻居去了工作，所以我大音乐。吃了早饭，然后我写了今天的计划，睡着了。我醒了，因为邻居回来了。我去洗澡。我的淋浴堵塞了，我想把那个马桶下面的东西。我过的是一种充满耻辱的生活。然后我想了吃什么的，但是天气很脏。我后悔我去了外面。我发现了一个箱子，我的粉丝给了我猕猴桃。我去了小零食店买了刮胡子。我就在餐厅吃了很油的菜。快乐回家。然后我开始了看很笨的电影。

【示例段落二】
输入：没上班的一天。今天被雷声吵醒。吃完早饭坐地铁去了一个挺漂亮的地方，结果走错方向走到一片沼泽地。本想回家但看到了路标，发现那其实是个公园。逛了会儿觉得太无聊就往回走。路上遇到一只猫带我去了个奇怪的地方，那里有好多仓鼠笼子，里面都是仓鼠。屋里还有只巨大的玩具熊，臭烘烘的。我坐在石头上反思人生，觉得这么折腾到底图啥。最后回家躺在椅子上放了个屁，吃着饭看《蟹老板》电影，觉得自己就像个寄居蟹。
输出：我在中国没有工作的天。今天我醒了，因为大音的雷声。我吃了早饭，坐了地铁。我到了在漂亮的地方，但是我去了对面，结果我到了什么的沼泽。我想了回家，但是我发现了路标，但是就是什么的公园。我探索了一下，但是太无聊了，我开始了回家。突然看见一只猫，它带我去奇怪的地方。我发现豚鼠的笼子，很多豚鼠。在房间中我看了肥胖的黄熊，很拉屎臭味。我在石头上坐了，问自己，我做这一切是为了什么。所以我回家了，在椅子上放屁了。我一边吃饭，一边看螃蟹的电影。我是寄居蟹。

【示例段落三】
输入：今天打车去了三个村游乐园。去游乐园是因为生活太空虚，需要一点刺激。玩了一个项目，明白了为什么开车不能看手机。发现了安全带的诱惑，把我固定住了。最后去漂流，水溅到脸上，衣服全湿了。很伤心，但还是决定去第二个漂流项目。然后后悔了，差点被水淹死，现在留下了心理创伤。鞋子全湿了，跟粉丝拍了照。回家了，很讨厌洗袜子。
输出：今天醒来后我打车了，我去了三个村。在游乐园，因为我有空虚的生活，我想要刺激。我明白了，为什么开车的时候不要看手机。我发现了安全的诱惑，他带我去在。最后的我去漂流，开始了漂流，最后水溅到我的脸上，我的衣服都湿了。我很伤心，虽然我决定了我去第二个漂流。然后我后悔比较多，我差不多淹死在水里，现在我有心理创伤。我的鞋子都湿了，我和粉丝做了拍照。回家了，我讨厌洗袜子。
`;

// LLM 翻译函数
async function translateWithGemini(apiKey: string, text: string, systemPrompt: string) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: "gemini-3.5-flash", systemInstruction: systemPrompt });
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
    const scored = (corpusArray as { input: string; output: string; embedding: number[] }[]).map(item => ({
      input: item.input, output: item.output,
      similarity: cosineSimilarity(queryVector, item.embedding)
    }));
    scored.sort((a, b) => b.similarity - a.similarity);
    topExamples = scored.slice(0, 2);
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
