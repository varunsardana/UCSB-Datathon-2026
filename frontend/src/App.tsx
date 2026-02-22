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

//Component imports
import Navbar from './components/Navbar';
import SidebarLeft from './components/SidebarLeft';
import DisasterMap from './components/DisasterMap';
import KeyMetrics from './components/KeyMetrics';
import AnalyticsTabs from './components/AnalyticsTabs';
import AdvisorChat from './components/AdvisorChat';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

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
