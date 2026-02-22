import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { Filter, Loader2, ChevronRight, AlertCircle } from 'lucide-react';

// FIPS state prefix → abbreviation (used to derive state from fips_code)
const FIPS_TO_STATE = {
  "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
  "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
  "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
  "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
  "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
  "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
  "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
  "55":"WI","56":"WY",
};

const STATE_NAMES = {
  AL:"Alabama",AK:"Alaska",AZ:"Arizona",AR:"Arkansas",CA:"California",
  CO:"Colorado",CT:"Connecticut",DE:"Delaware",DC:"D.C.",FL:"Florida",
  GA:"Georgia",HI:"Hawaii",ID:"Idaho",IL:"Illinois",IN:"Indiana",
  IA:"Iowa",KS:"Kansas",KY:"Kentucky",LA:"Louisiana",ME:"Maine",
  MD:"Maryland",MA:"Massachusetts",MI:"Michigan",MN:"Minnesota",MS:"Mississippi",
  MO:"Missouri",MT:"Montana",NE:"Nebraska",NV:"Nevada",NH:"New Hampshire",
  NJ:"New Jersey",NM:"New Mexico",NY:"New York",NC:"North Carolina",ND:"North Dakota",
  OH:"Ohio",OK:"Oklahoma",OR:"Oregon",PA:"Pennsylvania",RI:"Rhode Island",
  SC:"South Carolina",SD:"South Dakota",TN:"Tennessee",TX:"Texas",UT:"Utah",
  VT:"Vermont",VA:"Virginia",WA:"Washington",WV:"West Virginia",WI:"Wisconsin",WY:"Wyoming",
};

const DISASTER_LABELS = {
  biological:       'Biological',
  earthquake:       'Earthquake',
  fire:             'Wildfire',
  flood:            'Flood',
  freezing:         'Freezing',
  hurricane:        'Hurricane',
  severe_ice_storm: 'Severe Ice Storm',
  severe_storm:     'Severe Storm',
  snowstorm:        'Snowstorm',
  tornado:          'Tornado',
  tropical_storm:   'Tropical Storm',
  winter_storm:     'Winter Storm',
};

const DISASTER_COLORS = {
  biological:       'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300',
  earthquake:       'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300',
  fire:             'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300',
  flood:            'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
  freezing:         'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-300',
  hurricane:        'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300',
  severe_ice_storm: 'bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300',
  severe_storm:     'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300',
  snowstorm:        'bg-slate-100 text-slate-700 dark:bg-slate-700/30 dark:text-slate-300',
  tornado:          'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  tropical_storm:   'bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300',
  winter_storm:     'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300',
};

function getStateFromFips(fipsCode) {
  const prefix = String(fipsCode).padStart(5, '0').slice(0, 2);
  return FIPS_TO_STATE[prefix] ?? null;
}

// ─── SidebarLeft ──────────────────────────────────────────────────────────────
//
// Props:
//   isOpen            {boolean}
//   toggle            {() => void}
//   onRunPrediction   {({ disasterType, fipsCode, state }) => void}
//   isLoading         {boolean}
//
const SidebarLeft = ({ isOpen, toggle, onRunPrediction, isLoading = false }) => {
  const [width, setWidth] = useState(270);
  const isResizing = useRef(false);

  // Scenario data fetched from backend
  const [scenarios, setScenarios] = useState([]);   // raw list from /api/predict/scenarios
  const [fetchError, setFetchError] = useState(false);
  const [fetchingScenarios, setFetchingScenarios] = useState(true);

  // Derived filter state
  const [selectedDisaster, setSelectedDisaster] = useState(null);
  const [selectedState, setSelectedState]       = useState(null);
  const [selectedScenario, setSelectedScenario] = useState(null);

  // ── Fetch available scenarios from backend on mount ──────────────────────
  useEffect(() => {
    fetch('http://localhost:8000/api/predict/scenarios')
      .then(r => r.json())
      .then(data => {
        // Enrich each scenario with a derived state abbreviation
        const enriched = (data.scenarios ?? []).map(s => ({
          ...s,
          state: getStateFromFips(s.fips_code),
        }));
        setScenarios(enriched);
      })
      .catch(() => setFetchError(true))
      .finally(() => setFetchingScenarios(false));
  }, []);

  // ── Derived lists ─────────────────────────────────────────────────────────
  // Unique disaster types present in data
  const availableDisasters = [...new Set(scenarios.map(s => s.disaster_type))].sort();

  // States available for the selected disaster (or all states if none selected)
  const availableStates = [
    ...new Set(
      scenarios
        .filter(s => !selectedDisaster || s.disaster_type === selectedDisaster)
        .map(s => s.state)
        .filter(Boolean)
    )
  ].sort();

  // Filtered scenario list for the combo browser
  const filteredScenarios = scenarios.filter(s =>
    (!selectedDisaster || s.disaster_type === selectedDisaster) &&
    (!selectedState    || s.state         === selectedState)
  );

  // ── Resize logic ──────────────────────────────────────────────────────────
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      setWidth(Math.min(Math.max(e.clientX, 200), 420));
    };
    const handleMouseUp = () => { isResizing.current = false; };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleSelectDisaster = (type) => {
    setSelectedDisaster(prev => prev === type ? null : type);
    setSelectedState(null);
    setSelectedScenario(null);
  };

  const handleSelectState = (abbr) => {
    setSelectedState(prev => prev === abbr ? null : abbr);
    setSelectedScenario(null);
  };

  const handleSelectScenario = (scenario) => {
    setSelectedScenario(scenario);
  };

  const handleRun = () => {
    if (!selectedScenario && !selectedDisaster && !selectedState) return;
    onRunPrediction({
      disasterType: selectedScenario?.disaster_type ?? selectedDisaster ?? null,
      fipsCode:     selectedScenario?.fips_code     ?? null,
      state:        selectedScenario?.state         ?? selectedState ?? null,
    });
  };

  const canRun = (!!selectedScenario || !!selectedDisaster || !!selectedState) && !isLoading;

  return (
    <motion.aside
      initial={false}
      animate={{ width: isOpen ? width : 0, opacity: isOpen ? 1 : 0 }}
      className="h-full border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 overflow-hidden relative flex flex-shrink-0"
    >
      <div className="flex flex-col h-full overflow-hidden" style={{ width }}>

        {/* Header */}
        <div className="px-5 py-5 border-b border-slate-100 dark:border-slate-800 flex items-center gap-3 flex-shrink-0">
          <Filter size={18} className="text-primary" />
          <h3 className="font-bold text-slate-900 dark:text-white text-base">Scenario Browser</h3>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-5">

          {fetchingScenarios && (
            <div className="flex items-center gap-2 text-slate-400 text-xs">
              <Loader2 size={12} className="animate-spin" /> Loading scenarios...
            </div>
          )}

          {fetchError && (
            <div className="flex items-start gap-2 text-red-500 text-xs bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
              <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
              Could not load scenarios. Make sure the backend is running on port 8000.
            </div>
          )}

          {!fetchingScenarios && !fetchError && (
            <>
              {/* ── Step 1: Disaster type ──────────────────────────────── */}
              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                  1 · Disaster Type
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {availableDisasters.map(type => (
                    <button
                      key={type}
                      onClick={() => handleSelectDisaster(type)}
                      className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-all ${
                        selectedDisaster === type
                          ? `${DISASTER_COLORS[type] ?? 'bg-primary/10 text-primary'} border-transparent ring-2 ring-primary/30`
                          : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-slate-300'
                      }`}
                    >
                      {DISASTER_LABELS[type] ?? type}
                    </button>
                  ))}
                </div>
              </div>

              {/* ── Step 2: State (filtered by disaster) ──────────────── */}
              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                  2 · State{' '}
                  <span className="text-slate-300 font-normal normal-case tracking-normal">
                    ({availableStates.length} available)
                  </span>
                </p>
                {availableStates.length === 0 ? (
                  <p className="text-[11px] text-slate-400 italic">
                    Select a disaster type first
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {availableStates.map(abbr => (
                      <button
                        key={abbr}
                        onClick={() => handleSelectState(abbr)}
                        title={STATE_NAMES[abbr] ?? abbr}
                        className={`text-[11px] font-mono font-bold px-2 py-0.5 rounded border transition-all ${
                          selectedState === abbr
                            ? 'bg-primary text-white border-primary'
                            : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-primary/40 hover:text-primary'
                        }`}
                      >
                        {abbr}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* ── Step 3: Combo browser ──────────────────────────────── */}
              <div className="flex-1 min-h-0 flex flex-col">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2 flex-shrink-0">
                  3 · Scenario{' '}
                  <span className="text-slate-300 font-normal normal-case tracking-normal">
                    ({filteredScenarios.length} match{filteredScenarios.length !== 1 ? 'es' : ''})
                  </span>
                </p>
                <div className="space-y-1 flex-1 overflow-y-scroll pr-1">
                  {filteredScenarios.length === 0 && (
                    <p className="text-[11px] text-slate-400 italic">No scenarios match your filters</p>
                  )}
                  {filteredScenarios.map(s => {
                    const isSelected = selectedScenario?.key === s.key;
                    return (
                      <button
                        key={s.key}
                        onClick={() => handleSelectScenario(s)}
                        className={`w-full text-left px-3 py-2 rounded-lg border text-xs transition-all flex items-center justify-between gap-2 ${
                          isSelected
                            ? 'bg-primary/5 border-primary/30 text-primary'
                            : 'bg-slate-50 dark:bg-slate-800/50 border-slate-100 dark:border-slate-800 text-slate-700 dark:text-slate-300 hover:border-slate-200 hover:bg-white dark:hover:bg-slate-800'
                        }`}
                      >
                        <div className="min-w-0">
                          <p className="font-semibold truncate">{s.region ?? s.fips_code}</p>
                          <p className={`text-[10px] mt-0.5 ${isSelected ? 'text-primary/70' : 'text-slate-400'}`}>
                            {DISASTER_LABELS[s.disaster_type] ?? s.disaster_type}
                            {s.state ? ` · ${s.state}` : ''}
                          </p>
                        </div>
                        <ChevronRight size={12} className={`flex-shrink-0 ${isSelected ? 'text-primary' : 'text-slate-300'}`} />
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Selected summary */}
              {selectedScenario && (
                <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg px-3 py-2 text-xs border border-slate-100 dark:border-slate-800">
                  <p className="text-[10px] text-slate-400 uppercase font-bold mb-0.5">Selected</p>
                  <p className="font-semibold text-slate-800 dark:text-white truncate">{selectedScenario.region ?? selectedScenario.fips_code}</p>
                  <p className="text-slate-400 text-[10px]">{DISASTER_LABELS[selectedScenario.disaster_type] ?? selectedScenario.disaster_type}</p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Run button — pinned to bottom */}
        <div className="px-4 py-4 border-t border-slate-100 dark:border-slate-800 flex-shrink-0">
          <button
            onClick={handleRun}
            disabled={!canRun}
            className="w-full bg-primary text-white py-2.5 rounded-lg text-sm font-bold hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Running...
              </>
            ) : (
              'Run Prediction'
            )}
          </button>
          {!selectedScenario && !selectedDisaster && !selectedState && (
            <p className="text-[10px] text-slate-400 mt-2 text-center">Select a disaster type or state above</p>
          )}
        </div>
      </div>

      {/* Resize handle */}
      <div
        className="w-1 cursor-col-resize hover:bg-slate-300 active:bg-slate-400 flex-shrink-0"
        onMouseDown={() => { isResizing.current = true; }}
      />
    </motion.aside>
  );
};

export default SidebarLeft;