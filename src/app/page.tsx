"use client";

import { useState } from "react";
import "./globals.css";

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    
    setIsGenerating(true);
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      const data = await response.json();
      if (response.ok) {
        setOutputText(data.result);
      } else {
        setOutputText("【错误】: " + (data.error || "发生了未知的错误，我的老伙计。"));
      }
    } catch (error) {
      setOutputText("【网络错误】: 看起来网络被淹没了，我没法连接到服务器。");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main style={styles.main}>
      <div style={styles.header}>
        <h1 style={styles.title}>
          <span className="text-gradient">中译中</span> (英式/机翻中文) 转换器
        </h1>
        <p style={styles.subtitle}>体验那种充满耻辱的、荒谬的生活语录。写了你的正常中文，看看我的老伙计怎么说。</p>
      </div>

      <div style={styles.container}>
        {/* 输入区 */}
        <div className="glass-panel" style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={styles.badge}>正常中文 (索然无味)</span>
          </div>
          <textarea
            style={styles.textarea}
            placeholder="今天我休息没上班，睡了个懒觉。点了个外卖，真难吃。然后看了部烂片就睡觉了..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
        </div>

        {/* 转换按钮 */}
        <div style={styles.actionArea}>
          <button 
            style={{...styles.button, opacity: isGenerating ? 0.7 : 1}}
            onClick={handleTranslate}
            disabled={isGenerating}
          >
            {isGenerating ? "生成中..." : "立刻转换 ➔"}
          </button>
        </div>

        {/* 输出区 */}
        <div className="glass-panel" style={styles.panel}>
          <div style={styles.panelHeader}>
            <span style={{...styles.badge, background: 'rgba(147, 51, 234, 0.2)', color: '#d8b4fe'}}>机翻/英式中文 (充满抓马)</span>
          </div>
          <div style={styles.outputArea}>
            {outputText ? (
              <p style={styles.outputText}>{outputText}</p>
            ) : (
              <p style={styles.placeholder}>等待转换的文字将出现在这里，我的老伙计...</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

const styles = {
  main: {
    minHeight: '100vh',
    padding: '4rem 2rem',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '3rem',
  },
  header: {
    textAlign: 'center' as const,
    maxWidth: '600px',
  },
  title: {
    fontSize: '3rem',
    fontWeight: 800,
    marginBottom: '1rem',
    letterSpacing: '-0.02em',
  },
  subtitle: {
    fontSize: '1.1rem',
    color: 'rgba(255, 255, 255, 0.6)',
    lineHeight: 1.6,
  },
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2rem',
    width: '100%',
    maxWidth: '1000px',
  },
  panel: {
    display: 'flex',
    flexDirection: 'column' as const,
    padding: '1.5rem',
    gap: '1rem',
    minHeight: '250px',
  },
  panelHeader: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '0.5rem',
  },
  badge: {
    padding: '0.4rem 0.8rem',
    background: 'rgba(59, 130, 246, 0.2)',
    color: '#93c5fd',
    borderRadius: '20px',
    fontSize: '0.875rem',
    fontWeight: 600,
  },
  textarea: {
    width: '100%',
    flex: 1,
    background: 'transparent',
    border: 'none',
    color: 'var(--foreground)',
    fontSize: '1.1rem',
    lineHeight: 1.6,
    resize: 'none' as const,
    outline: 'none',
    fontFamily: 'var(--font-sans)',
  },
  outputArea: {
    flex: 1,
    padding: '0.5rem 0',
  },
  outputText: {
    fontSize: '1.1rem',
    lineHeight: 1.6,
    color: 'var(--foreground)',
  },
  placeholder: {
    fontSize: '1.1rem',
    color: 'rgba(255, 255, 255, 0.3)',
    fontStyle: 'italic',
  },
  actionArea: {
    display: 'flex',
    justifyContent: 'center',
    padding: '1rem 0',
  },
  button: {
    background: 'linear-gradient(135deg, var(--primary), #8b5cf6)',
    color: 'white',
    border: 'none',
    padding: '1rem 2.5rem',
    borderRadius: '100px',
    fontSize: '1.1rem',
    fontWeight: 600,
    cursor: 'pointer',
    boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)',
    transition: 'transform 0.2s, box-shadow 0.2s',
  }
};
