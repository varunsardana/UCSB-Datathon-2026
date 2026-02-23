import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

const FilteredAnalyticsTabs = () => {
  const [selectedState, setSelectedState] = useState(''); // Will hold selected state value
  const [selectedDisasterType, setSelectedDisasterType] = useState(''); // Will hold selected disaster type
  const [states, setStates] = useState([]); // States dropdown options
  const [disasterTypes, setDisasterTypes] = useState([]); // Disaster types dropdown options (state-dependent)
  
  // Loading states for dropdowns
  const [statesLoading, setStatesLoading] = useState(false);
  const [disasterTypesLoading, setDisasterTypesLoading] = useState(false);

  // Resizable chart height
  const [chartHeight, setChartHeight] = useState(350);
  const isResizingChart = useRef(false);
  const chartContainerRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingChart.current || !chartContainerRef.current) return;
      const rect = chartContainerRef.current.getBoundingClientRect();
      setChartHeight(Math.min(Math.max(e.clientY - rect.top, 200), 800));
    };
    const handleMouseUp = () => { isResizingChart.current = false; };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  // =======================================================================
  // 1. FETCH STATES FROM API
  // =======================================================================
  useEffect(() => {
    const fetchStates = async () => {
      setStatesLoading(true);
      try {
        const response = await fetch("http://localhost:8000/api/forecast/states");
        const data = await response.json();
        const stateOptions = [
            { value: '', label: 'All States' }, // Default option
            ...data.states.map(state => ({
            value: state,
            label: state
            }))
        ];
        setStates(stateOptions);
      } catch (error) {
        console.error('Failed to fetch states:', error);
      } finally {
        setStatesLoading(false);
      }
    };

    fetchStates();
  }, []);

  // =======================================================================
  // 2. FETCH DISASTER TYPES FROM API
  // =======================================================================
  // This runs whenever selectedState changes
  useEffect(() => {
    if (!selectedState) {
      setDisasterTypes([{ value: '', label: 'All Disaster Types' }]);
      setSelectedDisasterType('');
      return;
    }

    const fetchDisasterTypes = async () => {
      setDisasterTypesLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/api/forecast/types?state=${selectedState}`);
        const data = await response.json();
        
        const typeOptions = [
            { value: '', label: 'All Types' }, // Default option
            ...data.disaster_types.map(type => ({
            value: type,
            label: type
            }))
        ];
        
        setDisasterTypes(typeOptions);
      } catch (error) {
        console.error('Failed to fetch disaster types:', error);
        setDisasterTypes([{ value: '', label: 'Error loading types' }]);
      } finally {
        setDisasterTypesLoading(false);
      }
    };

    fetchDisasterTypes();
  }, [selectedState]);

  // =======================================================================  
// 3. FETCH CHART DATA
// =======================================================================
const [chartData, setChartData] = useState([]);
const [chartLoading, setChartLoading] = useState(false);
const [chartMeta, setChartMeta] = useState({});

useEffect(() => {
  if (!selectedState || !selectedDisasterType) {
    setChartData([]);
    return;
  }

  const fetchChartData = async () => {
    setChartLoading(true);
    try {
        const response = await fetch(`http://localhost:8000/api/forecast/chart?state=${selectedState}&disaster_type=${selectedDisasterType}`);
        const data = await response.json();
        
        // Monthly granularity — historical as line, forecast as area with confidence band
        const lastHist = data.historical[data.historical.length - 1];
        const combinedData = [
          ...data.historical.map(item => ({
            date: item.date,
            historical: item.count,
          })),
          // Bridge point so forecast connects to last historical value
          ...(lastHist ? [{
            date: lastHist.date,
            historical: lastHist.count,
            forecast: lastHist.count,
            confidence: [lastHist.count, lastHist.count],
          }] : []),
          ...data.forecast.map(item => ({
            date: item.date,
            forecast: Math.round(item.predicted * 100) / 100,
            confidence: [Math.round((item.lower ?? 0) * 100) / 100, Math.round(item.upper * 100) / 100],
          })),
        ];

        setChartData(combinedData);
        setChartMeta(data.meta);
    } catch (error) {
        console.error('Failed to fetch chart data:', error);
        setChartData([]);
    } finally {
        setChartLoading(false);
    }
  };
  fetchChartData();
  }, [selectedState, selectedDisasterType]);


  // Handler for state selection change
  const handleStateChange = (e) => {
    const newState = e.target.value;
    setSelectedState(newState);
    // Reset disaster type when state changes
    setSelectedDisasterType('');
  };

  // Handler for disaster type selection change
  const handleDisasterTypeChange = (e) => {
    setSelectedDisasterType(e.target.value);
  };

  // Log scale toggle
  const [logScale, setLogScale] = useState(false);

  // For log scale: shift zeros to a small value so log doesn't break
  const displayData = useMemo(() => {
    if (!logScale) return chartData;
    return chartData.map(d => ({
      ...d,
      historical: d.historical != null ? Math.max(d.historical, 0.1) : d.historical,
      forecast: d.forecast != null ? Math.max(d.forecast, 0.1) : d.forecast,
      confidence: d.confidence ? [Math.max(d.confidence[0], 0.1), Math.max(d.confidence[1], 0.1)] : d.confidence,
    }));
  }, [chartData, logScale]);

  const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const formatDate = (dateStr) => {
    if (!dateStr || !dateStr.includes('-')) return dateStr || '';
    const [year, month] = dateStr.split('-');
    return `${MONTH_NAMES[parseInt(month, 10) - 1]} ${year}`;
  };

  return (
    <div className="mt-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 px-6 py-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Historical Declaration Frequency</h2>
          <p className="text-xs text-slate-500">Prophet time-series forecast of FEMA disaster declarations</p>
        </div>

        {/* Inline filters + log toggle */}
        <div className="flex items-center gap-3">
          <div>
            <select
              value={selectedState}
              onChange={handleStateChange}
              disabled={statesLoading}
              className="px-3 py-1.5 border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-300 focus:ring-1 focus:ring-primary disabled:opacity-50"
            >
              {statesLoading ? (
                <option>Loading...</option>
              ) : (
                states.map((state) => (
                  <option key={state.value} value={state.value}>
                    {state.label}
                  </option>
                ))
              )}
            </select>
          </div>
          <div>
            <select
              value={selectedDisasterType}
              onChange={handleDisasterTypeChange}
              disabled={disasterTypesLoading || !selectedState}
              className="px-3 py-1.5 border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-300 focus:ring-1 focus:ring-primary disabled:opacity-50"
            >
              {disasterTypesLoading ? (
                <option>Loading...</option>
              ) : (
                disasterTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))
              )}
            </select>
          </div>
          <button
            onClick={() => setLogScale(prev => !prev)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              logScale
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400'
            }`}
          >
            {logScale ? 'Log' : 'Linear'}
          </button>
        </div>
      </div>

      <div ref={chartContainerRef} className="p-6" style={{ height: chartHeight }}>
        <div className="h-full flex flex-col">
            <div className="flex flex-col items-center justify-center h-full min-h-0">
            {chartLoading ? (
                <div className="text-slate-500 text-sm">Loading chart data...</div>
            ) : chartData.length === 0 ? (
                <div className="text-slate-500 text-sm">
                {selectedState && selectedDisasterType 
                    ? `No data for ${selectedState} - ${selectedDisasterType}` 
                    : 'Select state and disaster type to view frequency'
                }
                </div>
            ) : (
                <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={displayData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" strokeOpacity={0.3} />
                    <XAxis
                    dataKey="date"
                    tick={{ fontSize: 9, fill: '#94a3b8' }}
                    tickFormatter={(d) => d?.split?.('-')?.[0] ?? d}
                    interval={Math.max(Math.floor(displayData.length / 12), 1)}
                    />
                    <YAxis
                    scale={logScale ? 'log' : 'auto'}
                    domain={logScale ? [0.1, 'auto'] : [0, 'auto']}
                    allowDataOverflow={logScale}
                    label={{ value: `Declarations / month${logScale ? ' (log)' : ''}`, angle: -90, position: 'insideLeft', dy: 55, fontSize: 10, fill: '#94a3b8' }}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                    />
                    <Tooltip
                    labelFormatter={formatDate}
                    formatter={(value, name) => {
                      if (name === '95% Confidence') return [Array.isArray(value) ? `${value[0]} – ${value[1]}` : value, name];
                      return [`${value} declarations`, name];
                    }}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #334155', backgroundColor: '#1e293b', color: '#e2e8f0', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.3)' }}
                    labelStyle={{ color: '#94a3b8', marginBottom: 4 }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 11, paddingTop: 8, color: '#94a3b8' }} />

                    {/* Confidence band — shaded region between lower and upper */}
                    <Area
                        type="linear"
                        dataKey="confidence"
                        stroke="none"
                        fill="#3b82f6"
                        fillOpacity={0.15}
                        name="95% Confidence"
                        connectNulls={false}
                        isAnimationActive={false}
                    />

                    {/* Forecast — filled blue area */}
                    <Area
                    type="linear"
                    dataKey="forecast"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="#3b82f6"
                    fillOpacity={0.25}
                    name="Forecast"
                    connectNulls={false}
                    isAnimationActive={false}
                    />

                    {/* Historical — solid green line (monthly) */}
                    <Line
                    type="linear"
                    dataKey="historical"
                    stroke="#10b981"
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 4, fill: '#10b981' }}
                    name="Historical"
                    connectNulls={false}
                    isAnimationActive={false}
                    />
                </ComposedChart>
                </ResponsiveContainer>
            )}
            </div>
        </div>
      </div>
      {/* Resize handle */}
      <div
        className="h-3 cursor-row-resize hover:bg-primary/20 active:bg-primary/30 bg-primary/10 flex items-center justify-center gap-1 transition-colors group rounded-b-xl"
        onMouseDown={() => { isResizingChart.current = true; }}
      >
        <div className="w-8 h-[3px] rounded-full bg-primary/40 group-hover:bg-primary" />
      </div>
    </div>
  );
};

export default FilteredAnalyticsTabs;