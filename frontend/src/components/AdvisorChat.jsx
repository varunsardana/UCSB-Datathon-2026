import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

// Use relative URL — Vite proxies /api/* to http://localhost:8000 (see vite.config.ts)
const API_BASE = '';

// All 50 states + DC for the context selector
const US_STATES = [
  'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
  'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
  'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
  'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
];

// Disaster types the model supports (matches backend model_predictions keys)
const DISASTER_TYPES = [
  { value: 'hurricane',     label: 'Hurricane' },
  { value: 'flood',         label: 'Flood' },
  { value: 'fire',          label: 'Wildfire / Fire' },
  { value: 'severe_storm',  label: 'Severe Storm' },
  { value: 'tornado',       label: 'Tornado' },
  { value: 'earthquake',    label: 'Earthquake' },
  { value: 'biological',    label: 'Biological' },
];

const SUGGESTION_CHIPS = [
  'What happens to my job?',
  'Which sectors are hiring now?',
  'What aid programs can help me?',
  'How long until the job market recovers?',
];

const AdvisorChat = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Welcome to the DisasterShift Workforce Advisor.\n\n' +
        'I combine disaster frequency forecasts and employment impact models to give grounded, specific guidance.\n\n' +
        'Set your state and disaster type above, then ask me anything about job risk, recovery timelines, or available programs.',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedState, setSelectedState] = useState('');
  const [selectedDisasterType, setSelectedDisasterType] = useState('');
  const [showContext, setShowContext] = useState(true);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();

    // Append the user's message immediately
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsLoading(true);

    // Append an empty assistant placeholder — we'll stream tokens into it
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          state: selectedState || null,
          disaster_type: selectedDisasterType || null,
          job_title: null,
          fips_code: null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      // Read the SSE stream chunk by chunk
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Accumulate decoded bytes; stream:true handles multi-byte chars at chunk boundaries
        buffer += decoder.decode(value, { stream: true });

        // SSE events are delimited by double newlines
        const parts = buffer.split('\n\n');

        // The last element may be an incomplete event — hold it for the next chunk
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;

          const data = line.slice(6); // strip the "data: " prefix

          if (data === '[DONE]') {
            setIsLoading(false);
            return;
          }

          if (data.startsWith('[ERROR]')) {
            const errText = data.slice(7).trim();
            setMessages(prev => {
              const msgs = [...prev];
              msgs[msgs.length - 1] = {
                role: 'assistant',
                content: `⚠️ Advisor error: ${errText}`,
              };
              return msgs;
            });
            setIsLoading(false);
            return;
          }

          // The backend escapes real newlines inside tokens as \\n
          // so the SSE framing (which uses bare \n\n) doesn't break.
          // We restore them here before displaying.
          const token = data.replace(/\\n/g, '\n');

          setMessages(prev => {
            const msgs = [...prev];
            msgs[msgs.length - 1] = {
              role: 'assistant',
              content: msgs[msgs.length - 1].content + token,
            };
            return msgs;
          });
        }
      }
    } catch (err) {
      // Replace the empty placeholder with a user-facing error message
      setMessages(prev => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = {
          role: 'assistant',
          content:
            'Could not reach the advisor service. Make sure the backend is running on localhost:8000 and try again.',
        };
        return msgs;
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Build a short label for the header subtitle so the user sees active context at a glance
  const contextLabel =
    [selectedState, selectedDisasterType ? selectedDisasterType.replace('_', ' ') : '']
      .filter(Boolean)
      .join(' · ') || 'No context set — responses will be general';

  return (
    <aside className="flex w-full md:w-96 flex-col border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 shadow-xl h-full">

      {/* ── Header ── */}
      <div className="p-5 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-3 mb-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 text-primary">
            <Sparkles size={18} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-slate-900 dark:text-white">Workforce Advisor</h3>
            <p className="text-[10px] text-slate-500 truncate">{contextLabel}</p>
          </div>
          {/* Toggle context selectors open/closed */}
          <button
            onClick={() => setShowContext(v => !v)}
            className="text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Toggle context filters"
          >
            {showContext ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {/* ── Context selectors (collapsible) ── */}
        {showContext && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            {/* State picker */}
            <div>
              <label className="block text-[9px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                State
              </label>
              <select
                value={selectedState}
                onChange={e => setSelectedState(e.target.value)}
                className="w-full text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 text-slate-700 dark:text-slate-300 focus:ring-1 focus:ring-primary"
              >
                <option value="">Any state</option>
                {US_STATES.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* Disaster type picker */}
            <div>
              <label className="block text-[9px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                Disaster
              </label>
              <select
                value={selectedDisasterType}
                onChange={e => setSelectedDisasterType(e.target.value)}
                className="w-full text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 text-slate-700 dark:text-slate-300 focus:ring-1 focus:ring-primary"
              >
                <option value="">Any type</option>
                {DISASTER_TYPES.map(d => (
                  <option key={d.value} value={d.value}>{d.label}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ── Message list ── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4 custom-scrollbar">
        {messages.map((m, i) => (
          <div
            key={i}
            className={cn('flex flex-col gap-1', m.role === 'user' ? 'items-end' : 'items-start')}
          >
            <div
              className={cn(
                'max-w-[88%] p-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap',
                m.role === 'user'
                  ? 'bg-primary text-white rounded-tr-none'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-tl-none'
              )}
            >
              {m.content || (
                <span className="flex items-center gap-1.5 text-slate-400 italic">
                  <Loader2 size={12} className="animate-spin" />
                  Thinking…
                </span>
              )}
            </div>
            <span className="text-[9px] text-slate-400 font-medium px-2">
              {m.role === 'user' ? 'You' : 'Advisor'}
            </span>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* ── Input area ── */}
      <div className="p-5 pt-0">
        {/* Suggestion chips */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {SUGGESTION_CHIPS.map(chip => (
            <button
              key={chip}
              onClick={() => setInput(chip)}
              disabled={isLoading}
              className="text-[10px] font-bold text-slate-500 bg-slate-100 dark:bg-slate-800 px-2.5 py-1 rounded-full hover:bg-primary/10 hover:text-primary transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {chip}
            </button>
          ))}
        </div>

        {/* Text input + send button */}
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
            disabled={isLoading}
            placeholder={isLoading ? 'Advisor is responding…' : 'Ask about jobs, risk, or recovery…'}
            className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl py-3 pl-4 pr-12 text-sm focus:ring-1 focus:ring-primary focus:border-primary dark:text-white disabled:opacity-60 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="absolute right-2 top-1.5 h-8 w-8 bg-primary text-white rounded-lg flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </div>
      </div>
    </aside>
  );
};

export default AdvisorChat;
