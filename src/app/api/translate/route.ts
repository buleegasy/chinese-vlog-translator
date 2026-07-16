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

请严格遵守以下转换规则，模仿给定语料的风格：
1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有"我"、"我的"、"我们"或"他"。
2. 【强行过去式】：在过去的动作后机械地加上"了"（例如："我开始了看书"、"我做了地铁"、"我感觉了创伤后应激障碍"）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。若参考示例中已经对某个词进行了降维，必须直接使用示例中的降维结果。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用"然后"、"但是"、"因为"、"所以"串联毫无逻辑关联的琐事流水账。
6. 【抓马情感】：仅在用户原句带有明显负面情绪或属于长篇流水账时，才可插入一句过度严肃、悲观的翻译腔感叹（例如："我过的是一种充满耻辱的生活"、"我觉得我又存在主义危机"）。普通中性的日常句子严禁捏造这些词汇。
7. 【严控长度】：输出的句子数量必须与输入保持绝对一致。输入一句，输出一句；输入三句，输出三句，不多也不少。
8. 【信息完整保留】：输入中每一个细节（人物、动作、地点、时间、情绪）都必须出现在输出中。绝对不允许省略或合并原文中的任何信息。

下面是你必须学习和模仿的经典语料库示例（Few-shot Examples），它们定义了你的输出风格：

【示例段落一】：
输入（正常）：没工作的时候，今天没睡觉起得特早。听到邻居去上班了我就把音乐声开得老大。吃过早饭写了计划我就睡着了，直到邻居回来把我吵醒。我去洗澡结果淋浴堵了，想用皮搋子通一下。我真觉得我活得太失败了。后来想找点吃的，但空气实在太差。后悔出门了。路上捡了个箱子，粉丝还送了我猕猴桃。去便利店买了剃须刀。在餐厅吃了个特别油腻的菜，然后开开心心回家看无脑片。
输出（机翻）：我在中国没有工作的天。今天我起床了很早，因为我没有睡觉。我听听邻居去了工作，所以我大音乐。吃了早饭，然后我写了今天的计划，睡着了。我醒了，因为邻居回来了。我去洗澡。我的淋浴堵塞了，我想把那个马桶下面的东西。我过的是一种充满耻辱的生活。然后我想了吃什么的，但是天气很脏。我后悔我去了外面。我发现了一个箱子，我的粉丝给了我猕猴桃。我去了小零食店买了刮胡子。刮完屁股以后，我就在餐厅吃了很油的菜。快乐回家。然后我开始了看很笨的电影。

【示例段落二】：
输入（正常）：没上班的一天。今天被雷声吵醒。吃完早饭坐地铁去了一个挺漂亮的地方，结果走错方向走到一片沼泽地。本想回家但看到了路标，发现那其实是个公园。逛了会儿觉得太无聊就往回走。路上遇到一只猫带我去了个奇怪的地方，那里有好多仓鼠笼子，里面都是仓鼠。屋里还有只巨大的玩具熊，臭烘烘的。我坐在石头上反思人生，觉得这么折腾到底图啥。最后回家躺在椅子上放了个屁，吃着饭看《蟹老板》电影，觉得自己就像个寄居蟹。
输出（机翻）：我在中国没有工作的天。今天我醒了，因为大音的雷声。我吃了早饭，坐了地铁。我到了在漂亮的地方，但是我去了对面，结果我到了什么的沼泽。我想了回家，但是我发现了路标，但是就是什么的公园。我探索了一下，但是太无聊了，我开始了回家。突然看见一只猫，它带我去奇怪的地方。我发现豚鼠的笼子，很多豚鼠。在房间中我看了肥胖的黄熊，很拉屎臭味。我在石头上坐了，问自己，我做这一切是为了什么。所以我回家了，在椅子上放屁了。我一边吃饭，一边看螃蟹的电影。我是寄居蟹。

【示例段落三】：
输入（正常）：没上班。今天睡醒去KFC上个厕所，接着在共享单车旁边迷茫。换个角度看到一家小零食店，难以置信他们盖这么高的大楼居然只是为了卖零食。然后逛了家有趣的店，身上穿了辣条联名衣服。听闻中国年轻人无辣不欢，没想到做辣条工序这么复杂。我尝了一个，味道还行。继续瞎逛，大家都很开心，但是环境太嘈杂拥挤，我简直要得创伤后应激障碍了。于是我果断离开，回上海吃了顿中餐。
输出（机翻）：我在中国没有工作的天。今天醒来后我去了肯德基，上了厕所。然后我在自行车旁边都不知道怎么办。从其他的角色我看见了小零食店，我不相信他们建造了很大的楼为了卖小零食而已。我看到了很有意思的店，我穿了辣条T恤。听说很多年轻的人爱吃辣椒，原来做辣条比我想象太复杂。我吃了一个，不错了。然后我继续了探索，每个人都开心，但是声音非常大，没有个人空间，我感觉了创伤后应激障碍。所以我离开了，我回家了上海，吃了中式。

【示例段落四】：
输入（正常）：今天拍上班的视频。醒来开始干活，结果发现摄影器材进水坏了，只能干等。后来维修工人来修好监控面板，我顺便把袜子晾干。修好后继续开工，导演说今天表现不错，伙食和服装也都满意。我觉得我越来越懂中国人了，还跟他们讲趣事。工作了8小时，他们有荣誉墙，我也拍了照，照片被挂在其他优秀员工旁边。最后去便利店买东西坐地铁回家，累得不行。
输出（机翻）：我在中国上班的视频。今天醒来后我们开始了工作，然后我意识到了摄影宝淹没了。我们应该等。突然水桶的人到了，他们开始了干燥监控面板。我袜子晾干这样的。修好了以后我们开始了工作，老师说今天的工作很好的，没有什么的抱怨。衣服好看，吃饭也很好。我发现了我提高了我对中国人的水平。我跟他们说很有意思的故事。我们工作了8小时。他们有荣誉墙。我也做了拍照，我挂他在墙上的和其他的受人尊敬的人旁边。然后我去便利店，然后再出站。我累死了，我回家了。
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

    // 组装动态 System Prompt = 固定示例段落 + RAG 动态召回
    const systemPrompt = `
${BASE_SYSTEM_PROMPT_RULES}

以下是通过语义检索额外召回的、与当前输入最相关的补充参考（如果和上面的示例有重复，请以上面的为准）：
${topExamples.map((item, idx) => `
【补充参考 ${idx + 1}（相关度: ${(item.similarity * 100).toFixed(1)}%）】：
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
