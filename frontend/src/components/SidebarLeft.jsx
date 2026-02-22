import { useState, useEffect, useRef } from 'react';
import motion from 'motion/react';

const SidebarLeft = (isOpen) => {
  const [width, setWidth] = useState(280);
  const isResizing = useRef(false); // Whether the user is dragging or not
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      const newWidth = Math.min(Math.max(e.clientX, 200), 500); // Clamp between 200–500px
      setWidth(newWidth);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  return (
  <motion.aside 
    initial={false}
    animate={{ width: isOpen ? width : 0, opacity: isOpen ? 1 : 0 }}
    className="h-full border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 overflow-hidden relative flex"
  >
    <div className="p-6" style={{ width }}>
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
    <div
      className="w-1 cursor-col-resize hover:bg-slate-300 active:bg-slate-400"
      onMouseDown={() => {
        isResizing.current = true;
      }}
    />
  </motion.aside>
  );
};

export default SidebarLeft;