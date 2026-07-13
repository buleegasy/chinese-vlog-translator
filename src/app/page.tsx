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
      <div style={styles.container}>
        {/* 输入框 */}
        <textarea
          style={styles.textarea}
          placeholder="在此输入中文（如：今天加班好累）..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />

        {/* 转换按钮 */}
        <button 
          style={{
            ...styles.button, 
            opacity: isGenerating ? 0.6 : 1,
            cursor: isGenerating ? 'not-allowed' : 'pointer'
          }}
          onClick={handleTranslate}
          disabled={isGenerating}
        >
          {isGenerating ? "正在翻译..." : "转换"}
        </button>

        {/* 输出区 */}
        {outputText && (
          <div style={styles.outputContainer}>
            <p style={styles.outputText}>{outputText}</p>
          </div>
        )}
      </div>
    </main>
  );
}

const styles = {
  main: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem 1.5rem',
    background: '#ffffff',
  },
  container: {
    width: '100%',
    maxWidth: '480px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1.25rem',
  },
  textarea: {
    width: '100%',
    height: '160px',
    padding: '1.25rem',
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    background: '#f8fafc',
    color: '#0f172a',
    fontSize: '1.05rem',
    lineHeight: '1.6',
    outline: 'none',
    resize: 'none' as const,
    fontFamily: 'inherit',
    transition: 'all 0.2s ease',
    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.02)',
  },
  button: {
    width: '100%',
    padding: '1rem',
    borderRadius: '16px',
    border: 'none',
    background: '#000000',
    color: '#ffffff',
    fontSize: '1.05rem',
    fontWeight: 600,
    letterSpacing: '0.05em',
    transition: 'all 0.2s ease',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  },
  outputContainer: {
    width: '100%',
    padding: '1.25rem',
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    background: '#ffffff',
    minHeight: '80px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.02)',
  },
  outputText: {
    fontSize: '1.05rem',
    lineHeight: '1.6',
    color: '#0f172a',
    margin: 0,
    whiteSpace: 'pre-wrap' as const,
  }
};
