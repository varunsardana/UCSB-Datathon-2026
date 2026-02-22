import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

const FilteredAnalyticsTabs = () => {
  // Filter states
  const [activeTab, setActiveTab] = useState('curve');
  const [selectedState, setSelectedState] = useState(''); // Will hold selected state value
  const [selectedDisasterType, setSelectedDisasterType] = useState(''); // Will hold selected disaster type
  const [states, setStates] = useState([]); // States dropdown options
  const [disasterTypes, setDisasterTypes] = useState([]); // Disaster types dropdown options (state-dependent)
  
  // Loading states for dropdowns
  const [statesLoading, setStatesLoading] = useState(false);
  const [disasterTypesLoading, setDisasterTypesLoading] = useState(false);

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
        
        //Transform for Recharts - combine historical + forecast
        const combinedData = [
            ...data.historical.map(item => ({
            date: new Date(item.date).getTime(),
            count: item.count,
            type: 'historical'
            })),
            ...data.forecast.map(item => ({
            date: new Date(item.date).getTime(),
            count: item.predicted, // Use predicted as main line
            lower: item.lower,
            upper: item.upper,
            type: 'forecast'
            }))
        ];
        
        setChartData(combinedData);
        setChartMeta(data.meta);
    } catch (error) {
        console.error('Failed to fetch chart data:', error);
        setChartData({ historical: [], forecast: [] });
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
            {tab === 'curve' && 'State and Disaster Specific Analytics'}
          </button>
        ))}
      </div>

      {/* =====================================================
           FILTER DROPDOWNS
           ==================================================== */}
      <div className="p-6 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/30">
        <div className="flex flex-col sm:flex-row gap-4 max-w-md">
          {/* State Filter Dropdown */}
          <div className="flex-1">
            <label className="block text-xs font-bold uppercase text-slate-500 mb-1 tracking-wider">
              Filter by State
            </label>
            <select
              value={selectedState}
              onChange={handleStateChange}
              disabled={statesLoading}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-sm focus:ring-2 focus:ring-primary focus:border-transparent disabled:bg-slate-100 disabled:cursor-not-allowed"
            >
              {statesLoading ? (
                <option>Loading states...</option>
              ) : (
                states.map((state) => (
                  <option key={state.value} value={state.value}>
                    {state.label}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Disaster Type Filter Dropdown */}
          <div className="flex-1">
            <label className="block text-xs font-bold uppercase text-slate-500 mb-1 tracking-wider">
              Disaster Type
            </label>
            <select
              value={selectedDisasterType}
              onChange={handleDisasterTypeChange}
              disabled={disasterTypesLoading || !selectedState}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-sm focus:ring-2 focus:ring-primary focus:border-transparent disabled:bg-slate-100 disabled:cursor-not-allowed"
            >
              {disasterTypesLoading ? (
                <option>Loading types...</option>
              ) : (
                disasterTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>

      <div className="p-6 h-[400px]">
        {activeTab === 'curve' && (
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
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis 
                    dataKey="date" 
                    domain={[chartData[0]?.date || 0, new Date('2030-12-31').getTime()]}
                    label={{ value: 'Date', position: 'insideBottom', offset: -5, fontSize: 10 }}
                    tick={{ fontSize: 10 }}
                    />
                    <YAxis 
                    label={{ value: 'Declarations per Month', angle: -90, position: 'insideLeft', dy: 60, fontSize: 10 }}
                    tick={{ fontSize: 10 }}
                    />
                    <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                    />
                    
                    {/* Historical data - solid green line */}
                    <Line 
                    type="monotone" 
                    data={chartData}
                    dataKey={(entry) => entry.type === 'historical' ? entry.count : null}
                    stroke="#10b981" 
                    strokeWidth={3} 
                    dot={true}
                    activeDot={{ r: 6 }}
                    name="Historical"
                    isAnimationActive={false}
                    />

                    {/* Forecast data - dashed blue line */}
                    <Line 
                    type="monotone" 
                    data={chartData}
                    dataKey={(entry) => entry.type === 'forecast' ? entry.count : null}
                    stroke="#3b82f6" 
                    strokeWidth={3} 
                    strokeDasharray="6 6"
                    dot={false}
                    name="Forecast"
                    isAnimationActive={false}
                    />

                </LineChart>
                </ResponsiveContainer>
            )}
            </div>
        </div>
        )}
      </div>
    </div>
  );
};

export default FilteredAnalyticsTabs;