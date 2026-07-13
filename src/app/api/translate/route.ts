import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";

export const runtime = 'edge';

// 初始化 Gemini API (使用我们在 .env.local 中配置的 Key)
// 注意：如果您的环境确实叫 3.5 flash，可以替换模型名称，但目前最稳定且速度最快的是 gemini-1.5-flash
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

const SYSTEM_PROMPT = `
你现在是一个“机翻Vlogger语录”转换器。你需要将用户输入的正常中文，转换为一种生硬、滑稽、带有强烈英语母语者直译特征的“机翻中文/翻译腔”。

请严格遵守以下转换规则，模仿给定语料的风格：
1. 【疯狂加主语】：绝不省略主语。确保每个短句前面几乎都有“我”、“我的”、“我们”或“他”。
2. 【强行过去式】：在过去的动作后机械地加上“了”（例如：“我开始了看书”、“我做了地铁”、“我感觉了创伤后应激障碍”）。
3. 【词汇降维与直译】：不要使用高级或地道的中文词汇。把正常的词变成愚蠢的直译或描述（例如：放音乐 -> 大音乐；买剃须刀 -> 买了刮胡子；雾霾 -> 天气很脏；漏水 -> 淹没了；吃中餐 -> 吃了中式）。
4. 【语序倒装】：把表示程度的副词或时间状语放在句子最后（例如：今天我起床了很早；我开心了很大）。
5. 【逻辑生硬】：高频且机械地使用“然后”、“但是”、“因为”、“所以”串联毫无逻辑关联的琐事流水账。
6. 【抓马情感】：在流水账描述的最后或中间，突然插入一句过度严肃、悲观的翻译腔感叹（例如：“我过的是一种充满耻辱的生活”、“我觉得我又存在主义危机”、“我的人生彻底失败了”、“问自己我做这一切是为了什么”）。

不要输出任何解释，不要带有Markdown格式，直接输出转换后的一段话。
`;

export async function POST(req: Request) {
  try {
    const { text } = await req.json();

    if (!text) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 });
    }

    // 尝试使用 gemini-1.5-flash (如果您确认有更高的版本号，可以自行修改此处)
    const model = genAI.getGenerativeModel({ 
        model: "gemini-1.5-flash",
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
