import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, ChevronDown, ChevronUp, Loader2, Zap } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

// Use relative URL — Vite proxies /api/* to http://localhost:8000 (see vite.config.ts)
const API_BASE = '';

const US_STATES = [
  'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
  'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
  'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
  'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
];

const DISASTER_TYPES = [
  { value: 'hurricane',    label: 'Hurricane' },
  { value: 'flood',        label: 'Flood' },
  { value: 'fire',         label: 'Wildfire / Fire' },
  { value: 'severe_storm', label: 'Severe Storm' },
  { value: 'tornado',      label: 'Tornado' },
  { value: 'earthquake',   label: 'Earthquake' },
  { value: 'biological',   label: 'Biological' },
];

const SUGGESTION_CHIPS = [
  'Will I lose my job?',
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

  // Pipeline progress state
  const [statusSteps, setStatusSteps] = useState([]);
  const [liveSeconds, setLiveSeconds] = useState(0);
  const startTimeRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, statusSteps]);

  // Live elapsed timer while loading
  useEffect(() => {
    if (!isLoading) {
      setLiveSeconds(0);
      return;
    }
    const iv = setInterval(() => {
      if (startTimeRef.current) {
        setLiveSeconds(((Date.now() - startTimeRef.current) / 1000).toFixed(1));
      }
    }, 100);
    return () => clearInterval(iv);
  }, [isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsLoading(true);
    setStatusSteps([]);
    startTimeRef.current = Date.now();

    // Empty assistant placeholder — we'll stream into it
    setMessages(prev => [...prev, { role: 'assistant', content: '', metadata: null }]);

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
          audience_type: null, // auto-detected by backend from job title + question
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let contentStarted = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;

          const data = line.slice(6);

          if (data === '[DONE]') {
            const elapsed = ((Date.now() - startTimeRef.current) / 1000).toFixed(1);
            setMessages(prev => {
              const msgs = [...prev];
              msgs[msgs.length - 1] = {
                ...msgs[msgs.length - 1],
                metadata: { timeSeconds: elapsed },
              };
              return msgs;
            });
            setStatusSteps([]);
            setIsLoading(false);
            return;
          }

          if (data.startsWith('[ERROR]')) {
            const errText = data.slice(7).trim();
            setMessages(prev => {
              const msgs = [...prev];
              msgs[msgs.length - 1] = {
                role: 'assistant',
                content: `⚠️ ${errText}`,
                metadata: null,
              };
              return msgs;
            });
            setStatusSteps([]);
            setIsLoading(false);
            return;
          }

          // Pipeline status events — shown as progress steps, not message content
          if (data.startsWith('__status__')) {
            setStatusSteps(prev => [...prev, data.slice(10)]);
            continue;
          }

          // Real content token — start building the message
          contentStarted = true;
          const token = data.replace(/\\n/g, '\n');
          setMessages(prev => {
            const msgs = [...prev];
            msgs[msgs.length - 1] = {
              ...msgs[msgs.length - 1],
              content: msgs[msgs.length - 1].content + token,
            };
            return msgs;
          });
        }
      }
    } catch (err) {
      setMessages(prev => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = {
          role: 'assistant',
          content: 'Could not reach the advisor service. Make sure the backend is running on localhost:8000 and try again.',
          metadata: null,
        };
        return msgs;
      });
      setStatusSteps([]);
    } finally {
      setIsLoading(false);
    }
  };

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
          <button
            onClick={() => setShowContext(v => !v)}
            className="text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Toggle context filters"
          >
            {showContext ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {showContext && (
          <div className="mt-3 grid grid-cols-2 gap-2">
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
              {/* Loading state: show pipeline steps + live timer */}
              {m.content === '' && isLoading && i === messages.length - 1 ? (
                <div className="space-y-2 min-w-[180px]">
                  {statusSteps.length > 0 ? (
                    <div className="space-y-1.5">
                      {statusSteps.map((step, si) => (
                        <div
                          key={si}
                          className={cn(
                            'flex items-center gap-1.5 text-[10px] transition-all',
                            si < statusSteps.length - 1
                              ? 'text-emerald-500 dark:text-emerald-400'
                              : 'text-primary font-medium'
                          )}
                        >
                          <span>{si < statusSteps.length - 1 ? '✓' : '→'}</span>
                          <span>{step}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span className="flex items-center gap-1.5 text-slate-400 italic text-xs">
                      <Loader2 size={12} className="animate-spin" />
                      Starting…
                    </span>
                  )}
                  {/* Live elapsed timer */}
                  <div className="flex items-center gap-1 text-[9px] text-slate-400 pt-1 border-t border-slate-200 dark:border-slate-700">
                    <Loader2 size={9} className="animate-spin" />
                    {liveSeconds}s elapsed
                  </div>
                </div>
              ) : (
                m.content || (
                  <span className="flex items-center gap-1.5 text-slate-400 italic">
                    <Loader2 size={12} className="animate-spin" />
                    Thinking…
                  </span>
                )
              )}
            </div>

            {/* Message footer: role label + generation time badge */}
            <div className="flex items-center gap-2 px-2">
              <span className="text-[9px] text-slate-400 font-medium">
                {m.role === 'user' ? 'You' : 'Advisor'}
              </span>
              {m.metadata?.timeSeconds && (
                <span className="flex items-center gap-0.5 text-[9px] text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-full">
                  <Zap size={8} className="text-amber-400" />
                  {m.metadata.timeSeconds}s
                </span>
              )}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* ── Input area ── */}
      <div className="p-5 pt-0">
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

        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
            disabled={isLoading}
            placeholder={isLoading ? `Advisor is responding… ${liveSeconds}s` : 'Ask about jobs, risk, or recovery…'}
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
