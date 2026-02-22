import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';
import { X } from 'lucide-react';
//Below are imports for the map to work and look good
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';


const cn = (...inputs) => twMerge(clsx(inputs));

{ /*Map icon for disasters*/ }
const getSeverityIcon = (severity) => {
  const severityClass = severity === 'major' ? 'bg-red-500' : 'bg-orange-500';
  
  return new L.divIcon({
    html: `
      <div class="${severityClass} w-4 h-4 rounded-full border-2 border-white shadow-lg flex items-center justify-center">
        <div class="w-2 h-2 bg-white rounded-full" />
      </div>
    `,
    className: 'custom-div-icon bg-transparent border-0 p-0', // removes leaflet defaults
    iconSize: [15, 15],
    iconAnchor: [12, 24],
  });
};

const DisasterMap = () => {
  const [disasters, setDisasters] = useState([]);
  const [selectedDisaster, setSelectedDisaster] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/disasters")
      .then(res => res.json())
      .then(data => setDisasters(data));
  }, []);

  //Helper: fit map to disasters when they load
  const FitBoundsOnData = ({ points }) => {
    const map = useMap();

    useEffect(() => {
      if (!points || points.length === 0) return;

      const bounds = L.latLngBounds(
        points.map((p) => [p.lat, p.lng]),
      );
      map.fitBounds(bounds, { padding: [40, 40] });
    }, [points, map]);

    return null;
  };

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
            {disasters.length} Events Loaded
          </div>
        </div>
      </div>
      <div className="relative h-[400px] w-full bg-slate-100 dark:bg-slate-950 overflow-hidden">
        {/* Interactive map */}
        <MapContainer
          center={[37.8, -96]} // fallback center (roughly US)
          zoom={4}
          scrollWheelZoom={true}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Auto-fit to disasters when loaded */}
          {disasters.length > 0 && <FitBoundsOnData points={disasters} />}

          {/* Disaster markers */}
          {disasters.map((d) => (
            <Marker
              key={d.disaster_id}
              position={[d.lat, d.lng]}
              icon={getSeverityIcon(d.severity)}
              eventHandlers={{
                click: () => setSelectedDisaster(d),
              }}
            >
              {/* Popup on marker itself on click */}
              <Popup>
                <div className="text-sm">
                  <div className="font-semibold">{d.title}</div>
                  <div className="text-xs text-slate-500">
                    {d.state} • {d.county}
                  </div>
                  <div className="text-xs text-slate-500">
                    {d.declaration_date}
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>

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
              <h4 className="font-bold text-slate-900 dark:text-white">{selectedDisaster.title}</h4>
              <p className="text-xs text-slate-500 mb-3">{selectedDisaster.state} • {selectedDisaster.county} • {selectedDisaster.declaration_date}</p>
              
              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-100 dark:border-slate-800">
                {/* <div>
                  <p className="text-[10px] text-slate-400 uppercase font-bold">Displaced Workers</p>
                  <p className="text-sm font-bold text-primary">{selectedDisaster.displacedWorkers.toLocaleString()}</p>
                </div> */}
                {/* <div>
                  <p className="text-[10px] text-slate-400 uppercase font-bold">Main Industry</p>
                  <p className="text-sm font-bold text-slate-700 dark:text-slate-200">{selectedDisaster.mostAffectedIndustry}</p>
                </div> */}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Legend */}
        {/* <div className="absolute top-6 left-6 rounded-lg border border-slate-200 bg-white/80 p-3 shadow-sm backdrop-blur dark:border-slate-700 dark:bg-slate-900/80">
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
        </div> */}
      </div>
    </div>
  );
};

export default DisasterMap;