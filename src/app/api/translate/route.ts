import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";

export const runtime = 'edge';

const SYSTEM_PROMPT = `
你现在是一个“机翻Vlogger语录”转换器。你需要将用户输入的正常中文，转换为一种生硬、滑稽、带有强烈英语母语者直译特征的“机翻中文/翻译腔”。

请严格遵守以下转换规则，模仿给定语料的风格：
1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有“我”、“我的”、“我们”或“他”。
2. 【强行过去式】：在过去的动作后机械地加上“了”（例如：“我开始了看书”、“我做了地铁”、“我感觉了创伤后应激障碍”）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用“然后”、“但是”、“因为”、“所以”串联毫无逻辑关联的琐事流水账。
6. 【抓马情感】：在长文描述的最后或中间，突然插入一句过度严肃、悲观的翻译腔感叹（例如：“我过的是一种充满耻辱的生活”、“我觉得我又存在主义危机”、“我的人生彻底失败了”、“问自己我做这一切是为了什么”）。
7. 【严控长度】：**核心规则！输出的句子数量、段落结构和信息量必须与输入的原句保持绝对一致。如果用户只输入了一句简短的话，你必须且只能输出对应的一句机翻中文，绝不能自行脑补、扩充成一整篇流水账或多出其他句子。**

不要输出任何解释，不要带有Markdown格式，直接输出转换后的一段话。
`;

export async function POST(req: Request) {
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "API Key is missing in Cloudflare environment. 请确保已在 Cloudflare 控制台的 Environment Variables 中添加了 GEMINI_API_KEY，且值正确。" }, { status: 500 });
    }

    // 初始化 Gemini API (在请求内部初始化，防止 Edge Runtime 预加载导致环境变量为空)
    const genAI = new GoogleGenerativeAI(apiKey);

    // 使用 2026 年最新的 gemini-3.5-flash 模型
    const model = genAI.getGenerativeModel({ 
        model: "gemini-3.5-flash",
        systemInstruction: SYSTEM_PROMPT,
    });

    const result = await model.generateContent(text);
    const response = await result.response;
    const translatedText = response.text();

    return NextResponse.json({ result: translatedText });
  } catch (error) {
    console.error("Translation error:", error);
    return NextResponse.json({ error: "Failed to translate" }, { status: 500 });
  }
}
