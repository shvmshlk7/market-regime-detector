"use client";

import { useEffect, useRef, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, BarChart, Bar } from "recharts";
import gsap from "gsap";
import { 
  Activity, ShieldCheck, TrendingUp, DollarSign, Clock, X, Newspaper, 
  ChevronRight, RefreshCw, Cpu, Globe, Zap, AlertTriangle, BarChart3,
  Layers, Terminal, Search
} from "lucide-react";

// --- Types ---
interface ApiData {
  date: string;
  regime: string;
  probabilities: Record<string, number>;
  target_weights: Record<string, number>;
  latest_prices: Record<string, number>;
  historical_regimes_30d: any[];
  asset_regimes?: Record<string, string>;
  usd_inr_rate?: number;
}

interface GlobalVitals {
  vix: number;
  yield_spread: number;
  spy_rsi: number;
  spy_mom_1m: number;
  vol_ratio: number;
}

interface ModalData {
  ticker: string;
  reason: string;
  news: { title: string; url: string; publisher: string; pubDate: string }[];
}

export default function Home() {
  const [data, setData] = useState<ApiData | null>(null);
  const [vitals, setVitals] = useState<GlobalVitals | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currency, setCurrency] = useState<"USD" | "INR">("USD");
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [modalData, setModalData] = useState<ModalData | null>(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [bootSequence, setBootSequence] = useState(0); // 0-100 for boot animation
  const [mounted, setMounted] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);

  const fetchMainData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/live-status?t=" + Date.now().toString(), { cache: "no-store" });
      if (!res.ok) throw new Error("Backend offline");
      const json = await res.json();
      setData(json);
      return json;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  };

  const fetchVitals = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/global-vitals?t=" + Date.now().toString(), { cache: "no-store" });
      if (res.ok) setVitals(await res.json());
    } catch (e) {
      console.error("Vitals failed", e);
    }
  };

  useEffect(() => {
    setMounted(true);
    const init = async () => {
      // Simulate boot sequence for expert feel
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
          progress = 100;
          setBootSequence(100);
          clearInterval(interval);
          setLoading(false);
        } else {
          setBootSequence(Math.floor(progress));
        }
      }, 100);

      await Promise.all([fetchMainData(), fetchVitals()]);
    };
    init();

    // Auto-refresh every 60s
    const timer = setInterval(() => {
      fetchMainData();
      fetchVitals();
    }, 60000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!loading && data && containerRef.current) {
      const tl = gsap.timeline();
      tl.fromTo(
        ".stagger-item",
        { opacity: 0, x: -20, filter: "blur(10px)" },
        { opacity: 1, x: 0, filter: "blur(0px)", duration: 0.6, stagger: 0.08, ease: "power4.out" }
      );
      
      gsap.fromTo(
        ".flicker-on-load",
        { opacity: 0 },
        { opacity: 1, duration: 0.1, repeat: 5, yoyo: true }
      );
    }
  }, [loading, data]);

  const handleRowClick = async (ticker: string) => {
    setSelectedTicker(ticker);
    setModalLoading(true);
    setModalData(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/asset-details/${ticker}?t=${Date.now()}`, { cache: "no-store" });
      if (res.ok) setModalData(await res.json());
    } catch (e) {
      console.error(e);
    } finally {
      setModalLoading(false);
    }
  };

  const formatPrice = (usdPrice: number) => {
    if (currency === "INR" && data?.usd_inr_rate) {
      return `₹${(usdPrice * data.usd_inr_rate).toFixed(2)}`;
    }
    return `$${usdPrice.toFixed(2)}`;
  };

  if (!mounted || loading || bootSequence < 100) {
    return (
      <div className="flex flex-col h-screen w-full items-center justify-center bg-black text-primary font-mono crt-overlay">
        <div className="w-64 h-1 bg-surface-container-highest mb-4 overflow-hidden relative">
          <div className="h-full bg-primary transition-all duration-100" style={{ width: `${bootSequence}%` }} />
        </div>
        <div className="text-xs tracking-[0.3em] uppercase opacity-80 animate-flicker">
          Initializing Quant Kernel v4.2... {bootSequence}%
        </div>
        <div className="mt-8 text-[10px] text-on-surface-variant max-w-xs text-center opacity-50">
          Fetching neural weights from Gaussian HMM...<br />
          Optimizing portfolio matrix for current covariance...
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex h-screen items-center justify-center text-error bg-black crt-overlay font-mono">
        <AlertTriangle className="mr-4" />
        SYSTEM_FAILURE: {error || "DATA_LINK_OFFLINE"}
      </div>
    );
  }

  const pieData = Object.entries(data.target_weights).map(([key, value]) => ({
    name: key,
    value: value * 100,
  }));
  const COLORS = ["#98f040", "#4de082", "#a0f0d1", "#7ed321", "#5eb030", "#3a6d1d"];

  // Logic to handle potential mismatches in regime name capitalization or keys
  const getProb = (reg: string) => {
    if (!data.probabilities) return 0;
    // Check direct match
    if (data.probabilities[reg] !== undefined) return data.probabilities[reg];
    // Check case-insensitive
    const lowerReg = reg.toLowerCase();
    const entry = Object.entries(data.probabilities).find(([k]) => k.toLowerCase() === lowerReg);
    return entry ? entry[1] : 0;
  };

  const currentSureness = getProb(data.regime);

  return (
    <div className="min-h-screen bg-surface flex font-sans crt-overlay selection:bg-primary/30" ref={containerRef}>
      
      {/* --- Left Navigation / Vitals Sidebar --- */}
      <aside className="w-80 border-r border-outline-variant bg-surface-container-low flex flex-col hidden xl:flex overflow-y-auto custom-scrollbar stagger-item" ref={sidebarRef}>
        <div className="p-8 border-b border-outline-variant">
          <div className="flex items-center gap-3 mb-1">
             <div className="w-8 h-8 bg-primary rounded flex items-center justify-center text-surface animate-pulse-glow">
               <Layers size={18} strokeWidth={3} />
             </div>
             <h1 className="text-xl font-bold tracking-tighter text-white">COMMAND<span className="text-primary">DECK</span></h1>
          </div>
          <p className="text-[10px] font-mono text-on-surface-variant tracking-widest uppercase opacity-60">Quant Core Operating Environment</p>
        </div>

        <nav className="flex-1 p-6 space-y-8">
          {/* Global Vitals Section */}
          <div>
            <h3 className="text-[10px] font-bold text-outline-variant uppercase tracking-widest mb-6 flex items-center gap-2">
              <Globe size={12}/> Market Vitals
            </h3>
            <div className="space-y-4">
              {[
                { label: "VIX (Fear Index)", val: vitals?.vix, unit: "", color: "text-tertiary", icon: Activity },
                { label: "10Y-2Y Spread", val: vitals?.yield_spread, unit: "%", color: "text-secondary", icon: TrendingUp },
                { label: "SPY RSI (14d)", val: vitals?.spy_rsi, unit: "", color: "text-primary", icon: Zap },
                { label: "Vol Ratio (S/L)", val: vitals?.vol_ratio, unit: "x", color: "text-white", icon: Cpu }
              ].map((vital, i) => (
                <div key={i} className="bg-surface-container rounded-xl p-4 border border-outline-variant/30 card-hover-effect">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-on-surface-variant font-medium uppercase">{vital.label}</span>
                    <vital.icon size={12} className="opacity-40" />
                  </div>
                  <div className={`text-xl font-mono font-bold ${vital.color} flex items-baseline gap-1`}>
                    {vital.val ?? "---"}
                    <span className="text-[10px] opacity-60 ml-0.5">{vital.unit}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div className="p-4 rounded-xl border border-dashed border-outline-variant/50 bg-surface-container-low/50">
             <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[9px] text-outline-variant uppercase font-bold mb-1">Pipeline</p>
                  <p className="text-xs text-primary font-bold flex items-center gap-1">
                    <span className="w-1.5 h-1.5 bg-primary rounded-full animate-flicker" /> ACTIVE
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-outline-variant uppercase font-bold mb-1">Latency</p>
                  <p className="text-xs text-on-surface font-bold font-mono">42ms</p>
                </div>
             </div>
          </div>
        </nav>

        <div className="p-6 border-t border-outline-variant mt-auto">
          <button 
            onClick={() => setCurrency(currency === "USD" ? "INR" : "USD")}
            className="w-full py-3 bg-surface-container-high hover:bg-primary/20 hover:text-primary text-on-surface text-xs font-bold rounded-xl border border-outline-variant transition-all flex items-center justify-center gap-2 group"
          >
            <RefreshCw size={14} className="group-hover:rotate-180 transition-transform duration-500" />
             {currency === "USD" ? "VIEW IN INR (₹)" : "VIEW IN USD ($)"}
          </button>
        </div>
      </aside>

      {/* --- Main Stage --- */}
      <main className="flex-1 h-screen overflow-y-auto custom-scrollbar p-6 lg:p-10">
        
        {/* Top Header Grid */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10 stagger-item">
           <div className="relative">
              <div className="absolute -left-4 top-0 bottom-0 w-1 bg-primary rounded-full blur-[2px]" />
              <h2 className="text-3xl lg:text-5xl font-bold text-white tracking-tighter">
                ACTIVE <span className="text-primary text-glow">INTELLIGENCE</span>
              </h2>
              <p className="text-on-surface-variant font-mono text-sm mt-1 uppercase tracking-widest opacity-80">
                LATEST COMPUTE: {data.date} 
              </p>
           </div>
           
           <div className="flex gap-4">
              <div className="bg-surface-container-low px-5 py-3 rounded-2xl border border-outline-variant glass-panel hud-border min-w-[160px]">
                <p className="text-[10px] text-outline-variant uppercase tracking-widest font-bold mb-1">Global State</p>
                <p className="text-2xl font-bold flex items-center gap-2 text-white">
                   {data.regime === "Bull" ? <TrendingUp className="text-primary" /> : <Activity className="text-error" />}
                   {data.regime.toUpperCase()}
                </p>
              </div>
              <div className="bg-surface-container-low px-5 py-3 rounded-2xl border border-outline-variant glass-panel hud-border min-w-[160px]">
                <p className="text-[10px] text-outline-variant uppercase tracking-widest font-bold mb-1">Optimization</p>
                <p className="text-2xl font-bold text-white uppercase">{data.regime === "Bear" ? "HEDGE" : "MAX SHARPE"}</p>
              </div>
           </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-10">
          
          {/* Regime Confidence Gauge */}
          <div className="lg:col-span-4 bg-surface-container-low rounded-3xl border border-outline-variant p-8 glass-panel stagger-item relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16 group-hover:bg-primary/10 transition-colors" />
            <h3 className="text-xs font-bold text-outline-variant uppercase tracking-[0.2em] mb-8 flex items-center gap-2">
              <Search size={14}/> Confidence Metric
            </h3>
            
            <div className="h-64 relative flex items-center justify-center">
               <ResponsiveContainer width="100%" height="100%">
                 <PieChart>
                   <Pie
                     data={[
                       { name: 'Confidence', value: currentSureness * 100 },
                       { name: 'Remaining', value: (1 - currentSureness) * 100 }
                     ]}
                     cx="50%"
                     cy="85%"
                     startAngle={180}
                     endAngle={0}
                     innerRadius={80}
                     outerRadius={110}
                     paddingAngle={0}
                     dataKey="value"
                     stroke="none"
                   >
                     <Cell fill="var(--color-primary)" />
                     <Cell fill="rgba(65, 74, 54, 0.2)" />
                   </Pie>
                 </PieChart>
               </ResponsiveContainer>
               <div className="absolute inset-0 flex flex-col items-center justify-end pb-8">
                  <span className="text-5xl font-mono font-bold text-white leading-none">
                    {Math.round(currentSureness * 100)}%
                  </span>
                  <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest mt-2">Sureness Level</span>
               </div>
            </div>
            
            <div className="mt-4 grid grid-cols-3 gap-2">
               {Object.entries(data.probabilities || {}).map(([reg, prob], i) => (
                 <div key={i} className={`p-2 rounded-lg text-center border ${reg.toLowerCase() === data.regime.toLowerCase() ? 'border-primary/40 bg-primary/5' : 'border-outline-variant/30 opacity-50'}`}>
                    <p className="text-[8px] font-bold uppercase mb-0.5">{reg}</p>
                    <p className="text-xs font-mono font-bold text-white">{Math.round(prob * 100)}%</p>
                 </div>
               ))}
            </div>
          </div>

          {/* Main Chart Area */}
          <div className="lg:col-span-8 bg-surface-container-low rounded-3xl border border-outline-variant p-8 glass-panel stagger-item group">
            <div className="flex items-center justify-between mb-8">
               <h3 className="text-xs font-bold text-outline-variant uppercase tracking-[0.2em] flex items-center gap-2">
                <BarChart3 size={14}/> Regime Progression Matrix (30D)
              </h3>
              <div className="flex items-center gap-4 text-[10px] font-bold font-mono">
                 <span className="flex items-center gap-1.5 text-primary"><span className="w-2 h-2 rounded-full bg-primary" /> Bull</span>
                 <span className="flex items-center gap-1.5 text-on-surface-variant"><span className="w-2 h-2 rounded-full bg-on-surface-variant" /> Sideways</span>
                 <span className="flex items-center gap-1.5 text-error"><span className="w-2 h-2 rounded-full bg-error" /> Bear</span>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.historical_regimes_30d.map(item => ({
                  ...item,
                  regimeValue: item.regime === "Bull" ? 1 : item.regime === "Bear" ? -1 : 0
                }))}>
                  <defs>
                    <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <Tooltip 
                    contentStyle={{ background: '#0a140b', border: '1px solid #2d3c30', borderRadius: '12px', fontSize: '10px' }}
                    itemStyle={{ color: '#98f040' }}
                  />
                  <Area 
                    type="stepAfter" 
                    dataKey="regimeValue" 
                    stroke="var(--color-primary)" 
                    strokeWidth={2}
                    fill="url(#chartGrad)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Assets & Allocations Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-10 stagger-item">
          
          {/* Detailed Hold Table */}
          <div className="space-y-6">
            <h3 className="text-xl font-bold flex items-center gap-3 text-white">
               <Cpu className="text-primary"/> Target Allocation Matrix
            </h3>
            <div className="bg-surface-container-low border border-outline-variant rounded-3xl overflow-hidden glass-panel">
              <table className="w-full text-left border-collapse font-mono">
                <thead className="bg-surface-container-high/50 border-b border-outline-variant">
                  <tr className="text-[10px] text-outline-variant uppercase tracking-widest">
                    <th className="px-6 py-4 font-bold">Ticker</th>
                    <th className="px-6 py-4 font-bold">State</th>
                    <th className="px-6 py-4 font-bold text-right">Target Weight</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30">
                  {pieData.map((asset, i) => (
                    <tr key={asset.name} onClick={() => handleRowClick(asset.name)} className="data-table-row group cursor-pointer">
                      <td className="px-6 py-4">
                        <span className="text-lg font-bold text-white group-hover:text-primary transition-colors">{asset.name}</span>
                        <p className="text-[10px] text-on-surface-variant opacity-60">LP: {formatPrice(data.latest_prices[asset.name])}</p>
                      </td>
                      <td className="px-6 py-4">
                        <div className={`text-[10px] font-bold px-2 py-0.5 rounded border inline-block ${
                          data.asset_regimes?.[asset.name] === 'Bull' ? 'border-primary/20 text-primary bg-primary/5' :
                          data.asset_regimes?.[asset.name] === 'Bear' ? 'border-error/20 text-error bg-error/5' :
                          'border-outline-variant/20 text-on-surface-variant bg-surface-container'
                        }`}>
                          {data.asset_regimes?.[asset.name] || 'N/A'}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                         <div className="flex flex-col items-end gap-1">
                            <span className="text-lg font-bold text-white">{asset.value.toFixed(1)}%</span>
                            <div className="w-20 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                               <div className="h-full bg-primary" style={{ width: `${asset.value}%` }} />
                            </div>
                         </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Distribution Visualization */}
          <div className="space-y-6">
             <h3 className="text-xl font-bold flex items-center gap-3 text-white">
               <DollarSign className="text-primary"/> Diversification Profile
            </h3>
            <div className="bg-surface-container-low border border-outline-variant rounded-3xl p-8 glass-panel h-[410px] flex items-center justify-center">
               <ResponsiveContainer width="100%" height="100%">
                 <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={90}
                      outerRadius={125}
                      paddingAngle={8}
                      dataKey="value"
                      stroke="none"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#0a140b', border: 'none', borderRadius: '12px' }} />
                 </PieChart>
               </ResponsiveContainer>
               <div className="absolute flex flex-col items-center">
                  <span className="text-[10px] text-outline-variant font-bold uppercase tracking-widest">Total Position</span>
                  <span className="text-3xl font-bold text-white uppercase">UNIFIED</span>
               </div>
            </div>
          </div>

        </div>
      </main>

      {/* --- Global HUD Overlay Features --- */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex gap-4 pointer-events-none stagger-item opacity-40">
         <div className="flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 rounded-full border border-outline-variant text-[9px] font-mono font-bold tracking-tighter uppercase">
            <Terminal size={10} /> Kernel OK
         </div>
         <div className="flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 rounded-full border border-outline-variant text-[9px] font-mono font-bold tracking-tighter uppercase">
            <RefreshCw size={10} /> ML_SYNC_COMPLETE
         </div>
      </div>

      {/* Sliding Asset Context Menu */}
      <div 
        className={`fixed inset-y-0 right-0 w-full md:w-[540px] bg-surface-container-lowest border-l border-outline-variant shadow-2xl transform transition-transform duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] z-50 flex flex-col glass-panel-accent ${selectedTicker ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="p-10 border-b border-outline-variant flex items-center justify-between">
           <div>
              <p className="text-[10px] text-primary font-bold uppercase tracking-widest mb-1">Asset Intelligence</p>
              <h2 className="text-4xl font-bold text-white tracking-tighter uppercase font-mono">{selectedTicker}</h2>
           </div>
           <button onClick={() => setSelectedTicker(null)} className="w-12 h-12 flex items-center justify-center bg-surface-container-low hover:bg-error/20 hover:text-error rounded-full transition-all border border-outline-variant">
             <X size={24} />
           </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-10 space-y-12">
           {modalLoading ? (
             <div className="flex flex-col items-center justify-center h-full gap-4 opacity-40">
                <Activity className="animate-spin text-primary" size={48} />
                <p className="text-xs font-mono uppercase tracking-[0.3em]">Decoding Algorithmic State...</p>
             </div>
           ) : modalData && (
             <>
               <section>
                 <h3 className="text-xs font-bold text-outline-variant uppercase tracking-widest mb-6 flex items-center gap-2">
                   <Zap size={14} className="text-primary"/> Rational Analysis
                 </h3>
                 <div className="bg-surface-container p-6 rounded-2xl border border-outline-variant relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-5"><Cpu size={64}/></div>
                    <p className="text-white leading-relaxed text-sm font-medium relative z-10 italic">
                      "{modalData.reason}"
                    </p>
                 </div>
               </section>

               <section>
                 <h3 className="text-xs font-bold text-outline-variant uppercase tracking-widest mb-6 flex items-center gap-2">
                   <Newspaper size={14}/> Event Horizon (Live News)
                 </h3>
                 <div className="space-y-4">
                    {modalData.news.map((item, i) => (
                      <a key={i} href={item.url} target="_blank" rel="noopener noreferrer" className="block group bg-surface-container-low p-5 rounded-2xl border border-outline-variant hover:border-primary/50 transition-all card-hover-effect">
                         <div className="flex justify-between items-start mb-2 gap-4">
                           <h4 className="font-bold text-on-surface group-hover:text-primary transition-colors leading-tight">{item.title}</h4>
                           <ChevronRight size={16} className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                         </div>
                         <div className="flex items-center justify-between text-[10px] text-on-surface-variant font-bold uppercase tracking-widest">
                            <span>{item.publisher}</span>
                            <span>{item.pubDate}</span>
                         </div>
                      </a>
                    ))}
                 </div>
               </section>
             </>
           )}
        </div>
      </div>

       {/* Overlay */}
       {selectedTicker && (
        <div 
          className="fixed inset-0 bg-black/80 backdrop-blur-md z-40 transition-opacity duration-500"
          onClick={() => setSelectedTicker(null)}
        />
      )}

    </div>
  );
}
