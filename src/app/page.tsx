"use client";

import { useState } from "react";
import "./globals.css";

interface ReasoningItem {
  input: string;
  output: string;
  similarity: number;
}

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [reasoning, setReasoning] = useState<ReasoningItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    
    setIsGenerating(true);
    setReasoning([]);
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
        setReasoning(data.reasoning || []);
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
