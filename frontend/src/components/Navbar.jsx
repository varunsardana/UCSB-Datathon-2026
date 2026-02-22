import React from 'react';
import { Map as MapIcon, Info } from 'lucide-react';
import { APP_NAME } from '../constants';

const Navbar = () => (
  <header className="flex h-16 w-full items-center justify-between border-b border-slate-200 bg-white px-6 dark:border-slate-800 dark:bg-slate-900 z-50">
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-white">
        <MapIcon size={24} />
      </div>
      <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">{APP_NAME}</h1>
    </div>
    <nav className="hidden md:flex items-center gap-8">
      <a href="#" className="text-sm font-semibold text-primary border-b-2 border-primary py-5">Dashboard</a>
      <a href="#" className="text-sm font-medium text-slate-500 hover:text-primary py-5 transition-colors">About</a>
      <a href="#" className="text-sm font-medium text-slate-500 hover:text-primary py-5 transition-colors">Team</a>
      <button className="text-sm font-medium bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">Demo Script</button>
    </nav>
    <div className="flex items-center gap-4">
      <button className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
        <Info size={20} />
      </button>
      <div className="h-10 w-10 overflow-hidden rounded-full border-2 border-primary/20 bg-primary/10">
        <img 
          src="https://picsum.photos/seed/user/100/100" 
          alt="User" 
          className="h-full w-full object-cover"
          referrerPolicy="no-referrer"
        />
      </div>
    </div>
  </header>
);

export default Navbar;