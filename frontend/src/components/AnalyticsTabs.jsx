import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  BarChart,
  Bar
} from 'recharts';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';
import { MOCK_DISPLACEMENT_CURVE, MOCK_INDUSTRY_IMPACT } from '../mockData';

const cn = (...inputs) => twMerge(clsx(inputs));

const AnalyticsTabs = () => {
  const [activeTab, setActiveTab] = useState('curve');

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
      <div className="flex border-b border-slate-100 dark:border-slate-800">
        {['curve', 'impact', 'outlook'].map(tab => (
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

export default AnalyticsTabs;