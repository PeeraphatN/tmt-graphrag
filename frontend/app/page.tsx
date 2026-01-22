'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Trash2, Bot, User, Loader2, Copy, Check, Sparkles, Pill, Search, HelpCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types
interface Message {
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

// Example Questions
const EXAMPLE_QUESTIONS = [
  { icon: Pill, text: 'ยาในบัญชียาหลัก (NLEM) มีทั้งหมดกี่ตัว?' },
  { icon: Search, text: 'ใครผลิตยา Paracetamol?' },
  { icon: HelpCircle, text: 'ยา Amoxicillin อยู่ในบัญชียาหลักไหม?' },
];

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Copy to clipboard
  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Handle Send
  const handleSend = async (messageText?: string) => {
    const text = messageText || input.trim();
    if (!text || isLoading) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text, timestamp: new Date() }]);
    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) throw new Error('API Error');

      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'bot', content: data.response, timestamp: new Date() }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: '❌ เกิดข้อผิดพลาดในการเชื่อมต่อกับ Server', timestamp: new Date() },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Enter Key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Format Time
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <main className="flex flex-col h-screen bg-gradient-to-br from-slate-100 via-teal-50 to-emerald-100 text-slate-800 font-sans">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-lg border-b border-slate-200/50 p-4 shadow-sm z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="bg-gradient-to-br from-teal-500 to-emerald-600 p-2.5 rounded-xl text-white shadow-lg shadow-teal-200">
                <Bot size={24} />
              </div>
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
              />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-teal-600 to-emerald-600 bg-clip-text text-transparent">
                TMT Drug RAG
              </h1>
              <p className="text-xs text-slate-500">ระบบค้นหาข้อมูลยา</p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setMessages([])}
            className="flex items-center gap-2 text-slate-400 hover:text-red-500 transition-colors px-3 py-2 rounded-lg hover:bg-red-50"
          >
            <Trash2 size={18} />
            <span className="text-sm hidden sm:inline">ล้างประวัติ</span>
          </motion.button>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto pb-4">
          {/* Welcome Screen */}
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center mt-12 space-y-8"
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
              >
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-teal-100 text-teal-700 rounded-full text-sm font-medium">
                  <Sparkles size={16} />
                  Powered by Local LLM
                </div>
              </motion.div>
              <Bot size={80} className="mx-auto text-teal-300" />
              <div>
                <h2 className="text-3xl font-bold text-slate-700 mb-2">สวัสดีครับ! 👋</h2>
                <p className="text-slate-500 max-w-md mx-auto">
                  ถามข้อมูลยา TMT, ผู้ผลิต, ส่วนประกอบ หรือบัญชียาหลักแห่งชาติได้เลยครับ
                </p>
              </div>

              {/* Example Questions */}
              <div className="space-y-3">
                <p className="text-sm text-slate-400">ลองถามคำถามเหล่านี้:</p>
                <div className="flex flex-wrap justify-center gap-3">
                  {EXAMPLE_QUESTIONS.map((q, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.03, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSend(q.text)}
                      className="flex items-center gap-2 px-4 py-3 bg-white rounded-xl shadow-md hover:shadow-lg border border-slate-100 text-sm text-slate-600 transition-all"
                    >
                      <q.icon size={16} className="text-teal-500" />
                      <span>{q.text}</span>
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* Messages */}
          <AnimatePresence initial={false}>
            {messages.map((msg, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 15, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3 }}
                className={`flex gap-3 mb-6 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {/* Bot Avatar */}
                {msg.role === 'bot' && (
                  <div className="w-9 h-9 rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center text-white flex-shrink-0 mt-1 shadow-md">
                    <Bot size={18} />
                  </div>
                )}

                {/* Message Bubble */}
                <div className="group relative max-w-[80%]">
                  <div
                    className={`rounded-2xl px-5 py-3 shadow-md ${msg.role === 'user'
                      ? 'bg-gradient-to-r from-teal-500 to-emerald-600 text-white rounded-br-sm'
                      : 'bg-white/90 backdrop-blur-sm border border-slate-100 text-slate-800 rounded-bl-sm prose prose-sm max-w-none'
                      }`}
                  >
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  {/* Timestamp & Copy */}
                  <div className={`flex items-center gap-2 mt-1 text-[10px] text-slate-400 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <span>{formatTime(msg.timestamp)}</span>
                    {msg.role === 'bot' && (
                      <button
                        onClick={() => copyToClipboard(msg.content, index)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-slate-100 rounded"
                        title="คัดลอก"
                      >
                        {copiedIndex === index ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
                      </button>
                    )}
                  </div>
                </div>

                {/* User Avatar */}
                {msg.role === 'user' && (
                  <div className="w-9 h-9 rounded-full bg-slate-200 flex items-center justify-center text-slate-600 flex-shrink-0 mt-1 shadow-sm">
                    <User size={18} />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing Indicator */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3 items-start mb-6"
            >
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center text-white shadow-md">
                <Bot size={18} />
              </div>
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl rounded-bl-sm px-5 py-4 shadow-md border border-slate-100">
                <div className="flex gap-1.5">
                  <motion.div
                    className="w-2.5 h-2.5 bg-teal-400 rounded-full"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
                  />
                  <motion.div
                    className="w-2.5 h-2.5 bg-teal-400 rounded-full"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0.15 }}
                  />
                  <motion.div
                    className="w-2.5 h-2.5 bg-teal-400 rounded-full"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0.3 }}
                  />
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <footer className="bg-white/80 backdrop-blur-lg border-t border-slate-200/50 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="พิมพ์คำถามที่นี่..."
              className="flex-1 bg-white/80 backdrop-blur border border-slate-200 rounded-2xl pl-4 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-transparent resize-none shadow-sm text-slate-900 placeholder:text-slate-400 transition-all"
              rows={1}
              style={{ minHeight: '52px', maxHeight: '150px' }}
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              className="p-3 bg-gradient-to-r from-teal-500 to-emerald-600 text-white rounded-xl hover:from-teal-600 hover:to-emerald-700 disabled:opacity-50 disabled:hover:from-teal-500 disabled:hover:to-emerald-600 transition-all shadow-lg shadow-teal-200/50"
            >
              {isLoading ? <Loader2 className="animate-spin" size={22} /> : <Send size={22} />}
            </motion.button>
          </div>
          <p className="text-center text-[10px] text-slate-400 mt-2">
            🔒 ข้อมูลถูกประมวลผลภายในเครื่อง 100% | Local LLM + Neo4j
          </p>
        </div>
      </footer>
    </main>
  );
}
