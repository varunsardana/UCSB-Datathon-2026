import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  Filter, 
  Map as MapIcon, 
  BarChart3, 
  MessageSquare, 
  Info, 
  ChevronLeft, 
  ChevronRight,
  TrendingDown,
  TrendingUp,
  Clock,
  Users,
  Send,
  Sparkles,
  Menu,
  X
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend,
  BarChart,
  Bar,
  Cell,
  ReferenceLine
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { APP_NAME, PRIMARY_COLOR } from './constants';
import { 
  MOCK_DISASTERS, 
  MOCK_INDUSTRY_IMPACT, 
  MOCK_DISPLACEMENT_CURVE,
  type DisasterEvent,
  type IndustryImpact
} from './mockData';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Components ---

const Navbar = () => (
  <header className="flex h-16 w-full items-center justify-between border-b border-slate-200 bg-white px-6 dark:border-slate-800 dark:bg-slate-900 z-50">
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-white">
        <MapIcon size={24} />
      </div>
      <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">{APP_NAME}</h1>
    </div>
    <nav className="hidden md:flex items-center gap-8">
      <a href="#" className="text-sm font-semibold text-primary border-b-2 border-primary py-5">Dashboard</a>
      <a href="#" className="text-sm font-medium text-slate-500 hover:text-primary py-5 transition-colors">About</a>
      <a href="#" className="text-sm font-medium text-slate-500 hover:text-primary py-5 transition-colors">Team</a>
      <button className="text-sm font-medium bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">Demo Script</button>
    </nav>
    <div className="flex items-center gap-4">
      <button className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
        <Info size={20} />
      </button>
      <div className="h-10 w-10 overflow-hidden rounded-full border-2 border-primary/20 bg-primary/10">
        <img 
          src="https://picsum.photos/seed/user/100/100" 
          alt="User" 
          className="h-full w-full object-cover"
          referrerPolicy="no-referrer"
        />
      </div>
    </div>
  </header>
);

const SidebarLeft = ({ isOpen, toggle }: { isOpen: boolean, toggle: () => void }) => (
  <motion.aside 
    initial={false}
    animate={{ width: isOpen ? 280 : 0, opacity: isOpen ? 1 : 0 }}
    className="h-full border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 overflow-hidden relative"
  >
    <div className="p-6 w-[280px]">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Filter size={18} /> Scenario Inputs
        </h3>
      </div>
      
      <div className="space-y-6">
        <div>
          <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Disaster Type</label>
          <select className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm">
            <option>All Types</option>
            <option>Hurricane</option>
            <option>Wildfire</option>
            <option>Flood</option>
            <option>Earthquake</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Region</label>
          <div className="relative">
            <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
            <input 
              type="text" 
              placeholder="Search US States..." 
              className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg pl-10 pr-3 py-2 text-sm"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Severity Threshold</label>
          <input type="range" className="w-full accent-primary" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>Low</span>
            <span>Critical</span>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
          <button className="w-full bg-primary text-white py-2.5 rounded-lg text-sm font-bold hover:bg-primary/90 transition-colors">
            Run Prediction
          </button>
          <p className="text-[10px] text-slate-400 mt-2 text-center">
            {/* Backend API: POST /api/predict */}
            Updates charts based on scenario parameters
          </p>
        </div>
      </div>
    </div>
  </motion.aside>
);

const DisasterMap = () => {
  const [selectedDisaster, setSelectedDisaster] = useState<DisasterEvent | null>(MOCK_DISASTERS[0]);

  return (
    <div className="mb-6 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4 dark:border-slate-800">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Disaster Impact Map</h2>
          <p className="text-xs text-slate-500">Visualizing affected regions and labor disruption</p>
        </div>
        <div className="flex gap-2">
          <div className="text-xs font-medium px-3 py-1.5 bg-slate-100 dark:bg-slate-800 rounded-lg">
            {/* Backend API: GET /api/disasters */}
            {MOCK_DISASTERS.length} Events Loaded
          </div>
        </div>
      </div>
      <div className="relative h-[400px] w-full bg-slate-100 dark:bg-slate-950 overflow-hidden">
        {/* Placeholder for interactive map (e.g. Leaflet/Mapbox/D3) */}
        <div className="absolute inset-0 opacity-40 grayscale">
          <img 
            src="..\images\USMap.svg"
            alt="Map Background" 
            className="w-full h-full object-cover bg-white"
            referrerPolicy="no-referrer"
          />
        </div>
        
        {/* Mock Markers */}
        {MOCK_DISASTERS.map(d => (
          <motion.button
            key={d.id}
            whileHover={{ scale: 1.2 }}
            onClick={() => setSelectedDisaster(d)}
            className={cn(
              "absolute w-6 h-6 rounded-full border-2 border-white shadow-lg flex items-center justify-center transition-colors",
              d.severity === 'Critical' ? "bg-red-500" : "bg-orange-500"
            )}
            style={{ 
              left: `${(d.lng + 125) * 2}%`, 
              top: `${(50 - d.lat) * 4}%` 
            }}
          >
            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
          </motion.button>
        ))}

        {/* Tooltip / Info Overlay */}
        <AnimatePresence>
          {selectedDisaster && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="absolute bottom-6 left-6 right-6 md:right-auto md:w-80 rounded-xl border border-slate-200 bg-white/95 p-4 shadow-xl backdrop-blur dark:border-slate-700 dark:bg-slate-900/95 z-10"
            >
              <div className="flex justify-between items-start mb-2">
                <span className={cn(
                  "text-[10px] font-bold uppercase px-2 py-0.5 rounded",
                  selectedDisaster.severity === 'Critical' ? "bg-red-100 text-red-600" : "bg-orange-100 text-orange-600"
                )}>
                  {selectedDisaster.severity} Severity
                </span>
                <button onClick={() => setSelectedDisaster(null)} className="text-slate-400 hover:text-slate-600">
                  <X size={16} />
                </button>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white">{selectedDisaster.name}</h4>
              <p className="text-xs text-slate-500 mb-3">{selectedDisaster.region} • {selectedDisaster.date}</p>
              
              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-100 dark:border-slate-800">
                <div>
                  <p className="text-[10px] text-slate-400 uppercase font-bold">Displaced Workers</p>
                  <p className="text-sm font-bold text-primary">{selectedDisaster.displacedWorkers.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-400 uppercase font-bold">Main Industry</p>
                  <p className="text-sm font-bold text-slate-700 dark:text-slate-200">{selectedDisaster.mostAffectedIndustry}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Legend */}
        <div className="absolute top-6 left-6 rounded-lg border border-slate-200 bg-white/80 p-3 shadow-sm backdrop-blur dark:border-slate-700 dark:bg-slate-900/80">
          <h5 className="text-[10px] font-bold uppercase text-slate-500 mb-2">Job Loss Intensity</h5>
          <div className="flex items-center gap-1 h-2 w-32 rounded-full overflow-hidden">
            <div className="h-full w-1/4 bg-blue-100" />
            <div className="h-full w-1/4 bg-blue-300" />
            <div className="h-full w-1/4 bg-blue-500" />
            <div className="h-full w-1/4 bg-blue-700" />
          </div>
          <div className="flex justify-between text-[8px] text-slate-400 mt-1">
            <span>Low</span>
            <span>Critical</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const KeyMetrics = () => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    {[
      { label: 'Est. Displaced Workers', value: '12,400', icon: Users, trend: '+12%', color: 'text-primary' },
      { label: 'Peak Job Loss Month', value: 'Month 2', icon: TrendingDown, trend: 'Critical', color: 'text-red-500' },
      { label: 'Fastest-Growing Industry', value: 'Construction', icon: TrendingUp, trend: '+40%', color: 'text-emerald-500' },
      { label: 'Avg. Recovery Time', value: '8.4 Months', icon: Clock, trend: '-5%', color: 'text-amber-500' },
    ].map((m, i) => (
      <div key={i} className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm flex items-center justify-between">
        <div>
          <div className="flex items-center gap-1 mb-1">
            <p className="text-[10px] font-bold uppercase text-slate-400 tracking-wider">{m.label}</p>
            <Info size={10} className="text-slate-300 cursor-help" />
          </div>
          <h3 className={cn("text-xl font-bold", m.color)}>{m.value}</h3>
          <p className="text-[10px] font-medium text-slate-400 mt-1">
            <span className={cn(m.trend.startsWith('+') ? "text-emerald-500" : "text-slate-400")}>{m.trend}</span> vs baseline
          </p>
        </div>
        <div className="h-10 w-10 rounded-lg bg-slate-50 dark:bg-slate-800 flex items-center justify-center text-slate-400">
          <m.icon size={20} />
        </div>
      </div>
    ))}
  </div>
);

const AnalyticsTabs = () => {
  const [activeTab, setActiveTab] = useState<'curve' | 'impact' | 'outlook'>('curve');

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
      <div className="flex border-b border-slate-100 dark:border-slate-800">
        {(['curve', 'impact', 'outlook'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "px-6 py-4 text-sm font-bold transition-all border-b-2",
              activeTab === tab 
                ? "text-primary border-primary" 
                : "text-slate-400 border-transparent hover:text-slate-600"
            )}
          >
            {tab === 'curve' && 'Displacement Curve'}
            {tab === 'impact' && 'Industry Impact'}
            {tab === 'outlook' && 'Recovery Outlook'}
          </button>
        ))}
      </div>

      <div className="p-6 h-[400px]">
        {activeTab === 'curve' && (
          <div className="h-full flex flex-col">
            <div className="mb-4">
              <p className="text-sm font-medium text-slate-600 dark:text-slate-300">
                <span className="text-primary font-bold">Hospitality</span> drops 35% in first 2 months; 
                <span className="text-emerald-500 font-bold ml-1">Construction</span> demand peaks at Month 3.
              </p>
            </div>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={MOCK_DISPLACEMENT_CURVE}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis 
                    dataKey="month" 
                    label={{ value: 'Months relative to disaster', position: 'insideBottom', offset: -5, fontSize: 10 }}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis 
                    label={{ value: '% Change in Employment', angle: -90, position: 'insideLeft', fontSize: 10 }}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: 10, paddingTop: 10 }} />
                  <ReferenceLine x={0} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Disaster', position: 'top', fontSize: 10, fill: '#ef4444' }} />
                  <Line type="monotone" dataKey="Hospitality" stroke="#0284c5" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  <Line type="monotone" dataKey="Construction" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="Healthcare" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Retail" stroke="#6366f1" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'impact' && (
          <div className="h-full flex flex-col">
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={MOCK_INDUSTRY_IMPACT} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis dataKey="industry" type="category" tick={{ fontSize: 10 }} />
                  <Tooltip cursor={{ fill: 'transparent' }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="jobLossPct" name="Job Loss %" fill="#ef4444" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="demandIncreasePct" name="Demand Increase %" fill="#10b981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'outlook' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full overflow-y-auto custom-scrollbar pr-2">
            {MOCK_INDUSTRY_IMPACT.map((ind, i) => (
              <div key={i} className="border border-slate-100 dark:border-slate-800 rounded-xl p-4 bg-slate-50 dark:bg-slate-800/50 flex flex-col">
                <div className="flex justify-between items-start mb-3">
                  <h4 className="font-bold text-slate-900 dark:text-white">{ind.industry}</h4>
                  <span className={cn(
                    "text-[8px] font-bold uppercase px-2 py-0.5 rounded-full",
                    ind.riskScore === 'High' ? "bg-red-100 text-red-600" : 
                    ind.riskScore === 'Medium' ? "bg-amber-100 text-amber-600" : "bg-emerald-100 text-emerald-600"
                  )}>
                    {ind.riskScore} Risk
                  </span>
                </div>
                <div className="space-y-3 flex-1">
                  <div>
                    <p className="text-[10px] text-slate-400 uppercase font-bold">Expected Recovery</p>
                    <p className="text-xs font-medium text-slate-700 dark:text-slate-300">{ind.recoveryTime}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Recommended Pivot</p>
                    <div className="flex flex-wrap gap-1">
                      {ind.pivotSuggestions.map((s, j) => (
                        <span key={j} className="text-[9px] bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 px-2 py-0.5 rounded-md text-slate-600 dark:text-slate-300">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <button className="mt-4 w-full py-2 text-[10px] font-bold uppercase tracking-wider text-primary border border-primary/20 rounded-lg hover:bg-primary/5 transition-colors">
                  View Full Analysis
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const AdvisorChat = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Welcome to the Disaster Workforce Advisor. I\'ve analyzed the current scenario. How can I help you understand the labor market impact?' }
  ]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

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

// --- Main App Layout ---

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background-light dark:bg-background-dark">
      <Navbar />
      
      <main className="flex flex-1 overflow-hidden relative">
        {/* Mobile Sidebar Toggle */}
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="absolute left-4 bottom-4 z-50 md:hidden bg-primary text-white p-3 rounded-full shadow-lg"
        >
          {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        {/* Left Sidebar */}
        <SidebarLeft isOpen={isSidebarOpen} toggle={() => setIsSidebarOpen(!isSidebarOpen)} />

        {/* Center Content */}
        <div className="flex-1 flex flex-col overflow-y-auto p-6 custom-scrollbar">
          <div className="max-w-7xl mx-auto w-full">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Labor Market Dashboard</h2>
                <p className="text-sm text-slate-500">Real-time disruption analysis for <span className="text-primary font-bold">Hurricane Ian Scenario</span></p>
              </div>
              <div className="flex gap-3">
                <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium shadow-sm">
                  <Clock size={16} className="text-slate-400" /> History
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-bold shadow-md hover:bg-primary/90 transition-all">
                  Share Analysis
                </button>
              </div>
            </div>

            <DisasterMap />
            <KeyMetrics />
            <AnalyticsTabs />
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="hidden xl:block">
          <AdvisorChat />
        </div>
      </main>
    </div>
  );
}
