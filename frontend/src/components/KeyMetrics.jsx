import React from 'react';
import { Info, TrendingDown, TrendingUp, Clock, Users } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

const KeyMetrics = () => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    {[
      { label: 'Est. Displaced Workers', value: '12,400', icon: Users, trend: '+12%', color: 'text-primary' },
      { label: 'Peak Job Loss Month', value: 'Month 2', icon: TrendingDown, trend: 'Critical', color: 'text-red-500' },
      { label: 'Fastest-Growing Industry', value: 'Construction', icon: TrendingUp, trend: '+40%', color: 'text-emerald-500' },
      { label: 'Avg. Recovery Time', value: '8.4 Months', icon: Clock, trend: '-5%', color: 'text-amber-500' },
    ].map((m, i) => {
      const Icon = m.icon;
      return (
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
          <Icon size={20} />
        </div>
      </div>
      );
    })}
  </div>
);

export default KeyMetrics;