import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

const AdvisorChat = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome to the Disaster Workforce Advisor. I\'ve analyzed the current scenario. How can I help you understand the labor market impact?' }
  ]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    // Backend API: POST /api/chat
    // Simulate streaming response
    const assistantMsg = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, assistantMsg]);
    
    const fullResponse = "Based on the current Hurricane scenario, we expect a significant shift in demand towards Construction and Logistics. Hospitality workers in the coastal regions should consider immediate training in emergency logistics or claims processing, as these roles are seeing a 25% surge in demand.";
    
    let currentText = "";
    for (let i = 0; i < fullResponse.length; i++) {
      await new Promise(r => setTimeout(r, 10));
      currentText += fullResponse[i];
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = { role: 'assistant', content: currentText };
        return newMsgs;
      });
    }
  };

  return (
    <aside className="flex w-full md:w-96 flex-col border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 shadow-xl h-full">
      <div className="p-6 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-3 mb-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 text-primary">
            <Sparkles size={18} />
          </div>
          <h3 className="font-bold text-slate-900 dark:text-white">Workforce Advisor</h3>
        </div>
        <p className="text-[10px] text-slate-500">Ask about job risks, growth sectors, and training aid.</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
        {messages.map((m, i) => (
          <div key={i} className={cn("flex flex-col gap-1", m.role === 'user' ? "items-end" : "items-start")}>
            <div className={cn(
              "max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed",
              m.role === 'user' 
                ? "bg-primary text-white rounded-tr-none" 
                : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-tl-none"
            )}>
              {m.content || <span className="animate-pulse">...</span>}
            </div>
            <span className="text-[9px] text-slate-400 font-medium px-2">
              {m.role === 'user' ? 'You' : 'Advisor'}
            </span>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="p-6 pt-0">
        <div className="flex flex-wrap gap-2 mb-3">
          {['My job risk', 'Where are jobs growing?', 'Training options'].map(chip => (
            <button 
              key={chip} 
              onClick={() => setInput(chip)}
              className="text-[10px] font-bold text-slate-500 bg-slate-100 dark:bg-slate-800 px-3 py-1 rounded-full hover:bg-primary/10 hover:text-primary transition-all"
            >
              {chip}
            </button>
          ))}
        </div>
        <div className="relative">
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question..."
            className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl py-3 pl-4 pr-12 text-sm focus:ring-1 focus:ring-primary focus:border-primary dark:text-white"
          />
          <button 
            onClick={handleSend}
            className="absolute right-2 top-1.5 h-8 w-8 bg-primary text-white rounded-lg flex items-center justify-center hover:bg-primary/90 transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </aside>
  );
};

export default AdvisorChat;