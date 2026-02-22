import React, { useState } from 'react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Menu, X, Clock } from 'lucide-react';

import { APP_NAME } from './constants';

import Navbar from './components/Navbar';
import SidebarLeft from './components/SidebarLeft';
import DisasterMap from './components/DisasterMap';
import KeyMetrics from './components/KeyMetrics';
import AnalyticsTabs, { transformPrediction } from './components/AnalyticsTabs';
import AdvisorChat from './components/AdvisorChat';

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

// Shape of chart data passed to AnalyticsTabs
interface ChartData {
  displacementCurve: object[];
  industryImpact: object[];
}

interface PredictionMeta {
  region?: string;
  disasterType?: string;
}

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [chartData, setChartData] = useState<ChartData>({ displacementCurve: [], industryImpact: [] });
  const [predictionMeta, setPredictionMeta] = useState<PredictionMeta>({});
  const [error, setError] = useState<string | null>(null);

  // Called by SidebarLeft when the user hits "Run Prediction"
  const handleRunPrediction = async ({
    disasterType,
    fipsCode,
    state,
  }: {
    disasterType: string;
    fipsCode: string | null;
    state: string | null;
  }) => {
    setIsLoading(true);
    setHasRun(true);
    setError(null);

    try {
      let endpoint = 'http://localhost:8000/api/predict';
      let body: Record<string, string> = { disaster_type: disasterType };

      if (fipsCode) {
        body.fips_code = fipsCode;
      } else if (state) {
        endpoint = 'http://localhost:8000/api/predict/by-state';
        body.state = state;
      } else {
        setError('Please select a scenario from the list.');
        setIsLoading(false);
        return;
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`API error ${res.status}`);

      const data = await res.json();

      if (data.error) {
        setError('No prediction found for this scenario. Try a different combination.');
        setChartData({ displacementCurve: [], industryImpact: [] });
        return;
      }

      setChartData(transformPrediction(data));
      setPredictionMeta({
        region: data.region,
        disasterType: data.disaster_type,
      });
    } catch (err: any) {
      setError(err.message ?? 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const scenarioLabel = predictionMeta.region && predictionMeta.disasterType
    ? `${predictionMeta.disasterType.replace(/_/g, ' ')} — ${predictionMeta.region}`
    : 'Select a scenario to begin';

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

        {/* Left Sidebar — passes callbacks down */}
        <SidebarLeft
          isOpen={isSidebarOpen}
          toggle={() => setIsSidebarOpen(!isSidebarOpen)}
          onRunPrediction={handleRunPrediction}
          isLoading={isLoading}
        />

        {/* Center Content */}
        <div className="flex-1 flex flex-col overflow-y-auto p-6 custom-scrollbar">
          <div className="max-w-7xl mx-auto w-full">

            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Labor Market Dashboard</h2>
                <p className="text-sm text-slate-500">
                  Real-time disruption analysis for{' '}
                  <span className="text-primary font-bold capitalize">{scenarioLabel}</span>
                </p>
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

            {/* Error banner */}
            {error && (
              <div className="mb-4 px-4 py-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
                No prediction found for this scenario. Try a different disaster type or region.
              </div>
            )}

            <DisasterMap />
            <KeyMetrics />

            {/* AnalyticsTabs receives real data from the prediction */}
            <AnalyticsTabs
              displacementCurve={chartData.displacementCurve}
              industryImpact={chartData.industryImpact}
              isLoading={isLoading}
              hasRun={hasRun}
              region={predictionMeta.region}
              disasterType={predictionMeta.disasterType}
            />
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