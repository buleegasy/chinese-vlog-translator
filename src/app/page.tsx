"use client";

import { useState, useRef } from "react";
import "./globals.css";

interface ReasoningItem {
  input: string;
  output: string;
  similarity: number;
}

export default function Home() {
  // ⚡ Bolt Optimization: Replace controlled state with useRef for textarea.
  // This prevents the entire Home component from re-rendering on every single keystroke,
  // significantly improving typing performance, especially on lower-end devices.
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [outputText, setOutputText] = useState("");
  const [provider, setProvider] = useState("");
  const [reasoning, setReasoning] = useState<ReasoningItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleTranslate = async () => {
    const text = inputRef.current?.value || "";
    if (!text.trim()) return;
    
    setIsGenerating(true);
    setReasoning([]);
    setProvider("");
    setCopied(false);
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      if (response.ok) {
        setOutputText(data.result);
        setProvider(data.provider || "");
        setReasoning(data.reasoning || []);
      } else {
        setOutputText("【错误】: " + (data.error || "发生了未知的错误，我的老伙计。"));
      }
    } catch {
      setOutputText("【网络错误】: 看起来网络被淹没了，我没法连接到服务器。");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopy = async () => {
    if (!outputText) return;
    try {
      await navigator.clipboard.writeText(outputText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // 备用复制方法
      const textArea = document.createElement("textarea");
      textArea.value = outputText;
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err2) {
        console.error("Failed to copy", err2);
      }
      document.body.removeChild(textArea);
    }
  };

  return (
    <main style={styles.main}>
      <div style={styles.container}>
        {/* 输入框 */}
        <textarea
          ref={inputRef}
          style={styles.textarea}
          placeholder="在此输入中文（如：今天加班好累）..."
          defaultValue=""
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
            <div style={styles.outputHeader}>
              <button onClick={handleCopy} style={styles.copyButton}>
                {copied ? "已复制 ✓" : "复制结果"}
              </button>
            </div>
            <p style={styles.outputText}>{outputText}</p>
            {provider && (
              <div style={styles.providerInfo}>
                <span style={styles.providerBadge}>服务提供商: {provider}</span>
              </div>
            )}
          </div>
        )}

        {/* 推理展示区 */}
        {reasoning.length > 0 && (
          <div style={styles.reasoningSection}>
            <button 
              onClick={() => setShowReasoning(!showReasoning)} 
              style={styles.toggleButton}
            >
              {showReasoning ? "↓ 收起推理过程 (RAG)" : "→ 展开推理过程 (RAG)"}
            </button>

            {showReasoning && (
              <div style={styles.reasoningList}>
                <div style={styles.sectionHeader}>从知识库召回最接近的 2 条语料：</div>
                {reasoning.map((item, idx) => (
                  <div key={idx} style={styles.reasoningCard}>
                    <div style={styles.cardHeader}>
                      <span style={styles.exampleLabel}>参考示例 {idx + 1}</span>
                      <span style={styles.similarityBadge}>
                        语义相关度: {(item.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={styles.itemContent}>
                      <div style={styles.line}>
                        <span style={styles.lineLabel}>原始输入：</span>
                        <span style={styles.lineValue}>{item.input}</span>
                      </div>
                      <div style={styles.line}>
                        <span style={styles.lineLabel}>转换语气：</span>
                        <span style={styles.lineValue}>{item.output}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
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
    padding: '2.5rem 1.5rem',
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
    position: 'relative' as const,
    width: '100%',
    padding: '1.25rem',
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    background: '#ffffff',
    minHeight: '80px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.02)',
  },
  outputHeader: {
    display: 'flex',
    justifyContent: 'flex-end',
    marginBottom: '0.5rem',
  },
  copyButton: {
    background: 'none',
    border: '1px solid #e2e8f0',
    color: '#64748b',
    fontSize: '0.75rem',
    cursor: 'pointer',
    padding: '3px 8px',
    borderRadius: '6px',
    transition: 'all 0.15s ease',
    outline: 'none',
    fontWeight: 500,
  },
  outputText: {
    fontSize: '1.05rem',
    lineHeight: '1.6',
    color: '#0f172a',
    margin: 0,
    whiteSpace: 'pre-wrap' as const,
  },
  providerInfo: {
    marginTop: '0.875rem',
    borderTop: '1px solid #f1f5f9',
    paddingTop: '0.5rem',
  },
  providerBadge: {
    fontSize: '0.75rem',
    color: '#94a3b8',
    fontWeight: 500,
  },
  reasoningSection: {
    marginTop: '0.5rem',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.75rem',
  },
  toggleButton: {
    background: 'none',
    border: 'none',
    color: '#64748b',
    fontSize: '0.85rem',
    cursor: 'pointer',
    padding: '4px 0',
    textAlign: 'left' as const,
    fontWeight: 500,
    alignSelf: 'flex-start',
    outline: 'none',
  },
  reasoningList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
    padding: '1rem',
    borderRadius: '16px',
    background: '#f8fafc',
    border: '1px solid #f1f5f9',
  },
  sectionHeader: {
    fontSize: '0.85rem',
    color: '#475569',
    fontWeight: 600,
  },
  reasoningCard: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.5rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid #e2e8f0',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '0.8rem',
  },
  exampleLabel: {
    color: '#0f172a',
    fontWeight: 600,
  },
  similarityBadge: {
    color: '#059669',
    background: '#ecfdf5',
    padding: '2px 8px',
    borderRadius: '100px',
    fontWeight: 500,
  },
  itemContent: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.4rem',
    fontSize: '0.825rem',
    lineHeight: '1.5',
  },
  line: {
    display: 'flex',
    flexDirection: 'column' as const,
  },
  lineLabel: {
    color: '#64748b',
    fontWeight: 500,
    marginBottom: '2px',
  },
  lineValue: {
    color: '#334155',
  }
};
