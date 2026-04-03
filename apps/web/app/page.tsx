'use client';

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Bot,
  Check,
  Copy,
  HelpCircle,
  Loader2,
  Pill,
  Search,
  Send,
  Sparkles,
  Trash2,
  User,
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

interface Message {
  role: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const EXAMPLE_QUESTIONS = [
  { icon: Pill, text: 'ยา Paracetamol มีใครผลิตบ้าง?' },
  { icon: Search, text: 'ใครผลิตยา Amoxicillin?' },
  { icon: HelpCircle, text: 'ยา Simvastatin มีข้อมูลอะไรบ้าง?' },
];

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleSend = async (messageText?: string) => {
    const text = messageText || input.trim();
    if (!text || isLoading) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text, timestamp: new Date() }]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        throw new Error('API Error');
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'bot', content: data.response, timestamp: new Date() }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: 'เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <main className="flex h-screen flex-col bg-gradient-to-br from-slate-100 via-teal-50 to-emerald-100 text-slate-800">
      <header className="z-10 border-b border-slate-200/50 bg-white/80 p-4 shadow-sm backdrop-blur-lg">
        <div className="mx-auto flex max-w-4xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="rounded-xl bg-gradient-to-br from-teal-500 to-emerald-600 p-2.5 text-white shadow-lg shadow-teal-200">
                <Bot size={24} />
              </div>
              <motion.div
                className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-white bg-green-400"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
              />
            </div>
            <div>
              <h1 className="bg-gradient-to-r from-teal-600 to-emerald-600 bg-clip-text text-xl font-bold text-transparent">
                TMT Drug RAG
              </h1>
              <p className="text-xs text-slate-500">ระบบค้นหาข้อมูลยา</p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setMessages([])}
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-slate-400 transition-colors hover:bg-red-50 hover:text-red-500"
          >
            <Trash2 size={18} />
            <span className="hidden text-sm sm:inline">ล้างประวัติ</span>
          </motion.button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl pb-4">
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12 space-y-8 text-center"
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
              >
                <div className="inline-flex items-center gap-2 rounded-full bg-teal-100 px-4 py-2 text-sm font-medium text-teal-700">
                  <Sparkles size={16} />
                  Powered by Local LLM
                </div>
              </motion.div>
              <Bot size={80} className="mx-auto text-teal-300" />
              <div>
                <h2 className="mb-2 text-3xl font-bold text-slate-700">สวัสดีครับ</h2>
                <p className="mx-auto max-w-md text-slate-500">
                  ถามข้อมูลยา TMT, ผู้ผลิต, ส่วนประกอบ หรือบัญชียาหลักแห่งชาติได้เลยครับ
                </p>
              </div>

              <div className="space-y-3">
                <p className="text-sm text-slate-400">ลองถามคำถามเหล่านี้:</p>
                <div className="flex flex-wrap justify-center gap-3">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <motion.button
                      key={q.text}
                      whileHover={{ scale: 1.03, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSend(q.text)}
                      className="flex items-center gap-2 rounded-xl border border-slate-100 bg-white px-4 py-3 text-sm text-slate-600 shadow-md transition-all hover:shadow-lg"
                    >
                      <q.icon size={16} className="text-teal-500" />
                      <span>{q.text}</span>
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg, index) => (
              <motion.div
                key={`${msg.role}-${index}-${msg.timestamp.toISOString()}`}
                initial={{ opacity: 0, y: 15, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3 }}
                className={`mb-6 flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'bot' && (
                  <div className="mt-1 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 text-white shadow-md">
                    <Bot size={18} />
                  </div>
                )}

                <div className="group relative max-w-[80%]">
                  <div
                    className={`rounded-2xl px-5 py-3 shadow-md ${
                      msg.role === 'user'
                        ? 'rounded-br-sm bg-gradient-to-r from-teal-500 to-emerald-600 text-white'
                        : 'prose prose-sm max-w-none rounded-bl-sm border border-slate-100 bg-white/90 text-slate-800 backdrop-blur-sm'
                    }`}
                  >
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  <div
                    className={`mt-1 flex items-center gap-2 text-[10px] text-slate-400 ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <span>{formatTime(msg.timestamp)}</span>
                    {msg.role === 'bot' && (
                      <button
                        onClick={() => copyToClipboard(msg.content, index)}
                        className="rounded p-1 opacity-0 transition-opacity hover:bg-slate-100 group-hover:opacity-100"
                        title="คัดลอก"
                      >
                        {copiedIndex === index ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
                      </button>
                    )}
                  </div>
                </div>

                {msg.role === 'user' && (
                  <div className="mt-1 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-slate-200 text-slate-600 shadow-sm">
                    <User size={18} />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 flex items-start gap-3"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 text-white shadow-md">
                <Bot size={18} />
              </div>
              <div className="rounded-2xl rounded-bl-sm border border-slate-100 bg-white/90 px-5 py-4 shadow-md backdrop-blur-sm">
                <div className="flex gap-1.5">
                  <motion.div
                    className="h-2.5 w-2.5 rounded-full bg-teal-400"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
                  />
                  <motion.div
                    className="h-2.5 w-2.5 rounded-full bg-teal-400"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0.15 }}
                  />
                  <motion.div
                    className="h-2.5 w-2.5 rounded-full bg-teal-400"
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

      <footer className="border-t border-slate-200/50 bg-white/80 p-4 backdrop-blur-lg">
        <div className="mx-auto max-w-3xl">
          <div className="relative flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="พิมพ์คำถามที่นี่..."
              className="flex-1 resize-none rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 text-slate-900 shadow-sm transition-all placeholder:text-slate-400 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-teal-400"
              rows={1}
              style={{ minHeight: '52px', maxHeight: '150px' }}
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              className="rounded-xl bg-gradient-to-r from-teal-500 to-emerald-600 p-3 text-white shadow-lg shadow-teal-200/50 transition-all hover:from-teal-600 hover:to-emerald-700 disabled:opacity-50 disabled:hover:from-teal-500 disabled:hover:to-emerald-600"
            >
              {isLoading ? <Loader2 className="animate-spin" size={22} /> : <Send size={22} />}
            </motion.button>
          </div>
          <p className="mt-2 text-center text-[10px] text-slate-400">
            ข้อมูลถูกประมวลผลภายในเครื่อง | Local LLM + Neo4j
          </p>
        </div>
      </footer>
    </main>
  );
}
