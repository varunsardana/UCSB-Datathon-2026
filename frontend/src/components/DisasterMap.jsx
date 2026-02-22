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

export default DisasterMap;