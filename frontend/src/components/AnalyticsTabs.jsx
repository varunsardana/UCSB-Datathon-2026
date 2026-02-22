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
  Bar,
} from 'recharts';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

import { MOCK_DISPLACEMENT_CURVE, MOCK_INDUSTRY_IMPACT } from '../mockData';

const cn = (...inputs) => twMerge(clsx(inputs));

// ─── Sector colors ────────────────────────────────────────────────────────────
const SECTOR_COLORS = [
  '#0284c5', '#10b981', '#f59e0b', '#6366f1', '#ef4444',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#84cc16',
];
const getSectorColor = (i) => SECTOR_COLORS[i % SECTOR_COLORS.length];

// ─── Pivot suggestions by industry keyword ────────────────────────────────────
const PIVOT_SUGGESTIONS = {
  Hospitality:    ['Food Delivery', 'Catering', 'Property Mgmt'],
  Retail:         ['Warehousing', 'E-commerce', 'Logistics'],
  Construction:   ['Debris Removal', 'FEMA Contracting', 'Infrastructure'],
  Healthcare:     ['Telehealth', 'Crisis Response', 'Home Care'],
  Manufacturing:  ['Supply Chain', 'Gov Contracts', 'Repair Services'],
  Transportation: ['Emergency Logistics', 'Last-Mile Delivery', 'Trucking'],
  Education:      ['Online Tutoring', 'Workforce Training', 'EdTech'],
  Agriculture:    ['Food Bank Supply', 'Processing', 'Gov Relief Programs'],
  Finance:        ['Insurance Claims', 'Disaster Loans', 'Relief Admin'],
  Government:     ['Emergency Mgmt', 'FEMA Support', 'Public Works'],
};
const DEFAULT_PIVOTS = ['Retraining Programs', 'Federal Aid', 'Adjacent Sectors'];

function getPivots(industry) {
  const key = Object.keys(PIVOT_SUGGESTIONS).find(k =>
    industry.toLowerCase().includes(k.toLowerCase())
  );
  return key ? PIVOT_SUGGESTIONS[key] : DEFAULT_PIVOTS;
}

// ─── Risk score ───────────────────────────────────────────────────────────────
function getRiskScore(jobLossPct = 0, jobChangePct = 0) {
  if (jobLossPct > 25) return 'High';
  if (jobLossPct > 10) return 'Medium';
  if (jobChangePct > 50) return 'Low';
  return 'Low';
}

// ─── Synthesize month-by-month curve from flat prediction snapshot ────────────
const MONTHS = [-6, -3, 0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18];

function buildCurveForSector(data) {
  if ('job_loss_pct' in data) {
    const peak = -(data.job_loss_pct ?? 0);
    const recoveryMonth = data.recovery_months ?? 12;
    return MONTHS.map(m => {
      if (m < 0) return 0;
      if (m === 0) return peak;
      if (m >= recoveryMonth) return 0;
      return parseFloat((peak * (1 - m / recoveryMonth)).toFixed(1));
    });
  }
  if ('job_change_pct' in data) {
    const peak = data.job_change_pct ?? 0;
    const peakMonth = data.peak_month ?? 3;
    const decayEnd = peakMonth * 2 + 6;
    return MONTHS.map(m => {
      if (m < 0) return 0;
      if (m <= peakMonth) return parseFloat((peak * (m / peakMonth)).toFixed(1));
      if (m >= decayEnd) return 0;
      return parseFloat((peak * (1 - (m - peakMonth) / (decayEnd - peakMonth))).toFixed(1));
    });
  }
  return MONTHS.map(() => 0);
}

// ─── Transform /predict response → chart-ready data ──────────────────────────
export function transformPrediction(apiResponse) {
  if (!apiResponse || apiResponse.error || !apiResponse.predictions) {
    return { displacementCurve: [], industryImpact: [] };
  }

  const predictions = apiResponse.predictions;

  const industryImpact = Object.entries(predictions).map(([industry, data]) => ({
    industry,
    jobLossPct:        data.job_loss_pct   ?? 0,
    demandIncreasePct: data.job_change_pct ?? 0,
    riskScore:         getRiskScore(data.job_loss_pct, data.job_change_pct),
    recoveryTime:      data.recovery_months
                         ? `~${data.recovery_months} months`
                         : data.peak_month
                           ? `Peaks month ${data.peak_month}`
                           : 'Unknown',
    pivotSuggestions: getPivots(industry),
  }));

  const sectorCurves = {};
  for (const [industry, data] of Object.entries(predictions)) {
    sectorCurves[industry] = buildCurveForSector(data);
  }

  const displacementCurve = MONTHS.map((m, i) => {
    const point = { month: m };
    for (const [industry, curve] of Object.entries(sectorCurves)) {
      point[industry] = curve[i];
    }
    return point;
  });

  return { displacementCurve, industryImpact };
}

// ─── Fetch + transform helper (call this from your parent component) ──────────
export async function fetchAndTransform({ disasterType, fipsCode, severity = 'major' }) {
  const res = await fetch('/api/predict', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ disaster_type: disasterType, fips_code: fipsCode, severity }),
  });
  if (!res.ok) throw new Error(`Predict API error: ${res.status}`);
  return transformPrediction(await res.json());
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function EmptyState({ message = 'No prediction data available.' }) {
  return (
    <div className="h-full flex flex-col items-center justify-center gap-2 text-slate-400">
      <svg className="w-10 h-10 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9 17v-2m3 2v-4m3 4v-6M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
      </svg>
      <p className="text-sm">{message}</p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="h-full flex flex-col gap-4 animate-pulse">
      <div className="h-4 w-2/3 bg-slate-100 dark:bg-slate-800 rounded" />
      <div className="flex-1 bg-slate-100 dark:bg-slate-800 rounded-lg" />
    </div>
  );
}

// ─── AnalyticsTabs ────────────────────────────────────────────────────────────
//
// Props:
//   displacementCurve  {object[]}  — from transformPrediction(); falls back to mock
//   industryImpact     {object[]}  — from transformPrediction(); falls back to mock
//   isLoading          {boolean}
//   region             {string}    — e.g. "Los Angeles County, CA"
//   disasterType       {string}    — e.g. "wildfire"
//
const AnalyticsTabs = ({
  displacementCurve,
  industryImpact,
  isLoading = false,
  region,
  disasterType,
  hasRun = false,   // true once user has clicked Run Prediction at least once
}) => {
  const [activeTab, setActiveTab] = useState('curve');

  // Only show mock data if the user hasn't run a prediction yet.
  // After a run with no results, show empty state instead of misleading mock data.
  const showMock   = !hasRun;
  const curveData  = displacementCurve?.length ? displacementCurve : (showMock ? MOCK_DISPLACEMENT_CURVE : []);
  const impactData = industryImpact?.length    ? industryImpact    : (showMock ? MOCK_INDUSTRY_IMPACT    : []);
  const usingMock  = showMock && !displacementCurve?.length && !industryImpact?.length;

  const sectors = curveData.length ? Object.keys(curveData[0]).filter(k => k !== 'month') : [];

  const biggestLoss  = [...impactData].filter(d => d.jobLossPct > 0).sort((a, b) => b.jobLossPct - a.jobLossPct)[0];
  const biggestSurge = [...impactData].filter(d => d.demandIncreasePct > 0).sort((a, b) => b.demandIncreasePct - a.demandIncreasePct)[0];

  const tabs = ['curve', 'impact', 'outlook'];
  const tabLabels = { curve: 'Displacement Curve', impact: 'Industry Impact', outlook: 'Recovery Outlook' };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">

      {/* Tab bar */}
      <div className="flex border-b border-slate-100 dark:border-slate-800">
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              'px-6 py-4 text-sm font-bold transition-all border-b-2',
              activeTab === tab
                ? 'text-primary border-primary'
                : 'text-slate-400 border-transparent hover:text-slate-600',
            )}
          >
            {tabLabels[tab]}
          </button>
        ))}

        <div className="ml-auto flex items-center pr-4 gap-2">
          {disasterType && (
            <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-primary/10 text-primary">
              {disasterType.replace(/_/g, ' ')}
            </span>
          )}
          {region && <span className="text-[10px] text-slate-400 hidden md:block">{region}</span>}
          {usingMock && (
            <span className="text-[10px] font-medium text-amber-500 bg-amber-50 dark:bg-amber-900/20 px-2 py-0.5 rounded-full">
              Sample data
            </span>
          )}
        </div>
      </div>

      {/* Tab content */}
      <div className="p-6 h-[400px]">
        {isLoading ? <LoadingSkeleton /> : (
          <>
            {/* CURVE */}
            {activeTab === 'curve' && (
              <div className="h-full flex flex-col">
                <div className="mb-4">
                  <p className="text-sm font-medium text-slate-600 dark:text-slate-300">
                    {biggestLoss && (
                      <>
                        <span className="font-bold" style={{ color: getSectorColor(sectors.indexOf(biggestLoss.industry)) }}>
                          {biggestLoss.industry}
                        </span>
                        {' '}drops {biggestLoss.jobLossPct}% at disaster onset; recovery {biggestLoss.recoveryTime}.
                      </>
                    )}
                    {biggestSurge && (
                      <>
                        <span className="font-bold ml-1" style={{ color: getSectorColor(sectors.indexOf(biggestSurge.industry)) }}>
                          {biggestSurge.industry}
                        </span>
                        {' '}demand surges +{biggestSurge.demandIncreasePct}%.
                      </>
                    )}
                    {!biggestLoss && !biggestSurge && (
                      <span className="text-slate-400">Employment trajectory by sector over time.</span>
                    )}
                  </p>
                </div>
                {sectors.length === 0 ? (
                  <EmptyState message="No displacement curve data for this scenario." />
                ) : (
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={curveData}>
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
                          formatter={(v, name) => [`${v > 0 ? '+' : ''}${v}%`, name]}
                          contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                        />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: 10, paddingTop: 10 }} />
                        <ReferenceLine x={0} stroke="#ef4444" strokeDasharray="3 3"
                          label={{ value: 'Disaster', position: 'top', fontSize: 10, fill: '#ef4444' }} />
                        {sectors.map((sector, i) => (
                          <Line
                            key={sector}
                            type="monotone"
                            dataKey={sector}
                            stroke={getSectorColor(i)}
                            strokeWidth={i < 2 ? 3 : 2}
                            dot={i < 2 ? { r: 4 } : false}
                            activeDot={{ r: 6 }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* IMPACT */}
            {activeTab === 'impact' && (
              <div className="h-full flex flex-col">
                {impactData.length === 0 ? (
                  <EmptyState message="No industry impact data for this scenario." />
                ) : (
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={impactData} layout="vertical" margin={{ left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                        <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={v => `${v > 0 ? '+' : ''}${v}%`} />
                        <YAxis dataKey="industry" type="category" tick={{ fontSize: 10 }} width={90} />
                        <Tooltip cursor={{ fill: 'rgba(0,0,0,0.04)' }} formatter={(v, name) => [`${v > 0 ? '+' : ''}${v}%`, name]} />
                        <Legend wrapperStyle={{ fontSize: 10 }} />
                        <Bar dataKey="jobLossPct"        name="Job Loss %"        fill="#ef4444" radius={[0, 4, 4, 0]} />
                        <Bar dataKey="demandIncreasePct" name="Demand Increase %"  fill="#10b981" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* OUTLOOK */}
            {activeTab === 'outlook' && (
              impactData.length === 0 ? (
                <EmptyState message="No recovery outlook data for this scenario." />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full overflow-y-auto custom-scrollbar pr-2">
                  {impactData.map((ind, i) => (
                    <div key={i} className="border border-slate-100 dark:border-slate-800 rounded-xl p-4 bg-slate-50 dark:bg-slate-800/50 flex flex-col">
                      <div className="flex justify-between items-start mb-3">
                        <h4 className="font-bold text-slate-900 dark:text-white">{ind.industry}</h4>
                        <span className={cn(
                          'text-[8px] font-bold uppercase px-2 py-0.5 rounded-full',
                          ind.riskScore === 'High'   ? 'bg-red-100 text-red-600' :
                          ind.riskScore === 'Medium' ? 'bg-amber-100 text-amber-600' :
                                                       'bg-emerald-100 text-emerald-600',
                        )}>
                          {ind.riskScore} Risk
                        </span>
                      </div>

                      <div className="flex gap-3 mb-3">
                        {ind.jobLossPct > 0 && (
                          <div className="flex-1 bg-red-50 dark:bg-red-900/10 rounded-lg p-2 text-center">
                            <p className="text-[10px] text-red-400 font-bold uppercase">Job Loss</p>
                            <p className="text-sm font-bold text-red-600">{ind.jobLossPct}%</p>
                          </div>
                        )}
                        {ind.demandIncreasePct > 0 && (
                          <div className="flex-1 bg-emerald-50 dark:bg-emerald-900/10 rounded-lg p-2 text-center">
                            <p className="text-[10px] text-emerald-400 font-bold uppercase">Demand ↑</p>
                            <p className="text-sm font-bold text-emerald-600">+{ind.demandIncreasePct}%</p>
                          </div>
                        )}
                      </div>

                      <div className="space-y-3 flex-1">
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-bold">Expected Recovery</p>
                          <p className="text-xs font-medium text-slate-700 dark:text-slate-300">{ind.recoveryTime}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Recommended Pivot</p>
                          <div className="flex flex-wrap gap-1">
                            {(ind.pivotSuggestions ?? []).map((s, j) => (
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
              )
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default AnalyticsTabs;