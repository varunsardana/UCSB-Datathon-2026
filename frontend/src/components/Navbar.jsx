import React from 'react';
import { Map as MapIcon, Info } from 'lucide-react';
import { APP_NAME } from '../constants';

const Navbar = () => (
  <header className="flex h-15 w-full items-center justify-between border-b border-slate-200 bg-white px-6 dark:border-slate-800 dark:bg-slate-900 z-50">
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-white">
        <MapIcon size={22} />
      </div>
      <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">{APP_NAME}</h1>
    </div>
  </header>
);

export default Navbar;