import { useState, useRef, useEffect } from "react";
import Markdown from "react-markdown";
import "./App.css";

const API_URL = "https://vitalbioassistant.onrender.com/chat";
const SESSION_ID = crypto.randomUUID();

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: SESSION_ID }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Error: could not reach the server. Is the backend running on port 8000?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="header-icon">🧬</span>
          <div>
            <h1>VitalBio Assistant</h1>
            <p>OSHA 29 CFR § 1910.1030 — Bloodborne Pathogens</p>
          </div>
        </div>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask anything about bloodborne pathogen compliance.</p>
            <div className="suggestions">
              {[
                "What PPE is required when handling blood?",
                "What are the HBV vaccination requirements?",
                "How long must medical records be kept?",
                "What must an exposure control plan include?",
              ].map((q) => (
                <button
                  key={q}
                  className="suggestion"
                  onClick={() => setInput(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="bubble">
              {msg.role === "assistant" ? (
                <Markdown>{msg.content}</Markdown>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble loading">
              <span /><span /><span />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="input-bar">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a compliance question… (Enter to send)"
          rows={1}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={!input.trim() || loading}>
          Send
        </button>
      </footer>
    </div>
  );
}
