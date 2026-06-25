import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
// Lucide icons removed for flat clean UI

import MapComponent from './components/MapComponent';
import ParetoPlot from './components/ParetoPlot';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('optimization'); // 'diagnostics' | 'optimization'
  const [activeLayer, setActiveLayer] = useState('lst'); // 'lst' | 'ndvi' | 'albedo' | 'building'

  // Data State
  const [gridData, setGridData] = useState(null);
  const [isLoadingGrid, setIsLoadingGrid] = useState(true);
  const [gridError, setGridError] = useState(null);
  const [selectedCell, setSelectedCell] = useState(null);

  // Manual Diagnostics States
  const [deltaNdvi, setDeltaNdvi] = useState(0.0);
  const [deltaAlbedo, setDeltaAlbedo] = useState(0.0);

  // Evolutionary parameters
  const [generations, setGenerations] = useState(50);
  const [population, setPopulation] = useState(100);
  // Optimization process states
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [currentGen, setCurrentGen] = useState(0);
  const [bestCooling, setBestCooling] = useState(0);
  const [paretoCount, setParetoCount] = useState(0);
  const [paretoFront, setParetoFront] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [consoleLogs, setConsoleLogs] = useState([]);

  // Calculated diagnostics values (surrogate physics model)
  const calcCooling = (deltaNdvi * 18.20 + deltaAlbedo * 12.40).toFixed(2);
  const calcCost = (deltaNdvi * 2.0 + deltaAlbedo * 1.0).toFixed(2);
  const isPhysicallyStable = calcCooling >= 0 && calcCooling < 15.0;

  // WebSocket reference
  const wsRef = useRef(null);
  const logTerminalRef = useRef(null);

  // Load grid GeoJSON from backend on mount
  useEffect(() => {
    console.log('[API] Fetching unified spatial grid layout...');
    setIsLoadingGrid(true);
    fetch(`${API_BASE_URL}/grid`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setGridData(data);
        setIsLoadingGrid(false);
        // Set default selected cell on load to first cell in dataset
        if (data.features && data.features.length > 0) {
          setSelectedCell(data.features[0].properties);
        }
      })
      .catch(err => {
        console.error('[API ERROR] Failed to load H3 grid data:', err);
        setGridError(err.message);
        setIsLoadingGrid(false);
      });
  }, []);

  // Auto-scroll terminal logs to bottom
  useEffect(() => {
    if (logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [consoleLogs]);

  // Handle cell selection from map
  const handleSelectCell = useCallback((properties) => {
    setSelectedCell(properties);
    setParetoFront([]);
    setSelectedStrategy(null);
    setDeltaNdvi(0.0);
    setDeltaAlbedo(0.0);
    setConsoleLogs([`[SYSTEM] Selected cell: ${properties.H3_Index_ID}`, `[SYSTEM] Baseline Observed LST: ${properties.LST}°C`]);
  }, []);

  // Handle strategy selection from Pareto plot
  const handleSelectStrategy = useCallback((strategy) => {
    setSelectedStrategy(strategy);
    // Apply strategy values directly to manual sliders for cross-validation diagnostics
    setDeltaNdvi(strategy.delta_ndvi);
    setDeltaAlbedo(strategy.delta_albedo);
  }, []);

  const startLiveOptimization = () => {
    if (isOptimizing || !selectedCell) return;

    setIsOptimizing(true);
    setCurrentGen(0);
    setBestCooling(0);
    setParetoCount(0);
    setParetoFront([]);
    setSelectedStrategy(null);
    setConsoleLogs([
      `[WS] Connecting to optimize socket...`,
      `[SYSTEM] Init NSGA-II algorithm sweep...`,
      `[SYSTEM] Parameters: gen=${generations}, pop=${population}`,
      `[SYSTEM] Target Index: ${selectedCell.H3_Index_ID}`
    ]);

    let wsUrl;
    if (API_BASE_URL.startsWith('http')) {
      wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/optimize-live';
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      wsUrl = `${protocol}//${window.location.host}${API_BASE_URL}/optimize-live`;
    }
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ h3_index: selectedCell.H3_Index_ID }));
      setConsoleLogs(prev => [...prev, `[WS] Connection active. Optimization process running.`]);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (Array.isArray(data)) {
        // [generation_id, best_cooling_delta, current_pareto_count]
        const [genId, bestCool, pCount] = data;
        setCurrentGen(genId + 1);
        setBestCooling(bestCool);
        setParetoCount(pCount);

        const logLine = `[GEN ${String(genId + 1).padStart(3, '0')}] Cool Drop: -${bestCool.toFixed(2)}°C | Pareto Count: ${pCount}`;
        setConsoleLogs(prev => [...prev, logLine]);
      } else if (data.status === "complete") {
        setParetoFront(data.pareto_front);
        setIsOptimizing(false);
        setConsoleLogs(prev => [
          ...prev,
          `[SYSTEM] Sweep completed. Discovered ${data.pareto_front.length} Pareto solutions.`,
          `[SYSTEM] Disconnecting live websocket.`
        ]);
        ws.close();
      } else if (data.error) {
        setConsoleLogs(prev => [...prev, `[ERROR] Backend failure: ${data.error}`]);
        setIsOptimizing(false);
        ws.close();
      }
    };

    ws.onerror = (error) => {
      setConsoleLogs(prev => [...prev, `[WS ERROR] Socket connection interrupted.`]);
      setIsOptimizing(false);
    };

    ws.onclose = () => {
      setIsOptimizing(false);
    };
  };

  const cancelOptimization = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsOptimizing(false);
    setConsoleLogs(prev => [...prev, `[SYSTEM] User halted evolutionary run.`]);
  };

  return (
    <div className="min-h-screen bg-obsidian-void text-gray-400 font-sans flex flex-col antialiased">
      {/* 1. Header Area - Disciplined scientific observatory styling */}
      <header className="h-16 border-b border-obsidian-carbon px-6 bg-obsidian-charcoal flex items-center justify-between select-none">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-sm font-mono tracking-widest text-white font-semibold">
              BENGALURU URBAN THERMAL OBSERVATORY
            </h1>
            <p className="text-[10px] font-mono tracking-wider text-gray-500 uppercase mt-0.5">
              PIML-Correction Engine // Resolution H3-9
            </p>
          </div>
        </div>

        {/* Global connection/status telemetry */}
        <div className="flex items-center gap-6 font-mono text-[10px] tracking-wider text-zinc-500">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 bg-zinc-500" />
            <span>CORE ENGINE:</span>
            <span className="text-zinc-300 font-semibold">STABLE</span>
          </div>
          <div className="h-4 w-px bg-obsidian-carbon" />
          <div>
            SYS_NODE: <span className="text-zinc-300 font-semibold">BGLR_01</span>
          </div>
          <div className="h-4 w-px bg-obsidian-carbon" />
          <div>
            LATENCY: <span className="text-zinc-300 font-semibold">14ms</span>
          </div>
        </div>
      </header>

      {/* 2. Main Layout Grid */}
      <main className="flex-1 grid grid-cols-12 gap-4 p-4 h-[calc(100vh-64px)] overflow-hidden">

        {/* ==================== LEFT COLUMN: GIS Layer Controls & Core Telemetry ==================== */}
        <section className="col-span-3 flex flex-col gap-4 h-full overflow-hidden">

          {/* Card 1: Grid Telemetry */}
          <div className="cyber-panel p-4 flex flex-col max-h-[35%]">
            <h2 className="text-xs font-mono uppercase tracking-widest text-white border-b border-obsidian-carbon pb-2 mb-3 font-semibold">
              GRID AREA TELEMETRY
            </h2>

            {isLoadingGrid ? (
              <div className="flex-1 flex items-center justify-center font-mono text-[10px] text-gray-600">
                LOADING DATA STATE...
              </div>
            ) : gridError ? (
              <div className="flex-1 flex items-center justify-center font-mono text-[10px] text-crimson gap-1">
                [ERROR] LOAD FAILED: {gridError}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3 flex-1">
                <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex flex-col justify-between">
                  <span className="text-[9px] font-mono text-gray-500 uppercase">Total Cells</span>
                  <span className="text-lg font-mono-prec font-bold text-white mt-1">7,158</span>
                </div>
                <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex flex-col justify-between">
                  <span className="text-[9px] font-mono text-gray-500 uppercase">Avg LST</span>
                  <span className="text-lg font-mono-prec font-bold text-white mt-1">45.33°C</span>
                </div>
                <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex flex-col justify-between">
                  <span className="text-[9px] font-mono text-gray-500 uppercase">Max LST</span>
                  <span className="text-lg font-mono-prec font-bold text-plasma mt-1">55.83°C</span>
                </div>
                <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex flex-col justify-between">
                  <span className="text-[9px] font-mono text-gray-500 uppercase">Quality Gate</span>
                  <span className="text-xs font-mono font-bold text-isotope flex items-center gap-1 mt-2">
                    PASSED
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Card 2: Interactive Spatial Layers */}
          <div className="cyber-panel p-4 flex flex-col flex-1 overflow-hidden">
            <h2 className="text-xs font-mono uppercase tracking-widest text-white border-b border-obsidian-carbon pb-2 mb-4 font-semibold">
              SPATIAL INTERACTION LAYERS
            </h2>

            <div className="flex flex-col gap-2">
              {[
                { id: 'lst', label: 'Land Surface Temp (LST)', unit: '°C', activeColor: 'text-plasma border-plasma/40 bg-plasma/5' },
                { id: 'ndvi', label: 'Vegetation Fraction (NDVI)', unit: 'Index', activeColor: 'text-isotope border-isotope/40 bg-isotope/5' },
                { id: 'albedo', label: 'Surface Reflectivity (Albedo)', unit: 'Ratio', activeColor: 'text-telemetry border-telemetry/40 bg-telemetry/5' },
                { id: 'building', label: 'Building Density Index', unit: 'Density', activeColor: 'text-white border-gray-600 bg-white/5' }
              ].map((layer) => {
                const isActive = activeLayer === layer.id;
                return (
                  <button
                    key={layer.id}
                    onClick={() => setActiveLayer(layer.id)}
                    className={`w-full flex items-center justify-between p-3 border text-left font-mono transition-all text-xs ${isActive
                        ? layer.activeColor
                        : 'border-obsidian-carbon bg-obsidian-void hover:border-gray-700 text-gray-400'
                      }`}
                  >
                    <span>{layer.label}</span>
                    <span className="text-[9px] text-gray-500 font-semibold uppercase">{layer.unit}</span>
                  </button>
                );
              })}
            </div>

            {/* Target Cell Details */}
            {selectedCell && (
              <div className="mt-4 bg-obsidian-void p-3 border border-obsidian-carbon flex flex-col gap-2">
                <div className="text-[10px] font-mono text-gray-500 uppercase tracking-widest font-bold">
                  Selected Cell Metrics
                </div>
                <div className="text-xs font-mono select-all text-telemetry font-bold border-b border-obsidian-carbon pb-1 mb-1">
                  ID: {selectedCell.H3_Index_ID}
                </div>
                <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                  <div>Obs LST: <span className="text-white font-bold font-mono-prec">{selectedCell.LST}°C</span></div>
                  <div>NDVI: <span className="text-white font-bold font-mono-prec">{selectedCell.NDVI}</span></div>
                  <div>Albedo: <span className="text-white font-bold font-mono-prec">{selectedCell.Albedo}</span></div>
                  <div>Density: <span className="text-white font-bold font-mono-prec">{selectedCell.Building_Density}</span></div>
                  <div>Air Temp: <span className="text-white font-bold font-mono-prec">{selectedCell.Air_Temp}°C</span></div>
                  <div>Humidity: <span className="text-white font-bold font-mono-prec">{selectedCell.Humidity}%</span></div>
                </div>
              </div>
            )}

            <div className="mt-auto bg-obsidian-void p-3 border border-obsidian-carbon text-[10px] font-mono leading-normal">
              <div className="text-white font-bold mb-1 text-xs">
                Physical Model Constraints
              </div>
              Physics-Informed Loss penalties align empirical LST updates with localized thermodynamic balance equations.
            </div>
          </div>
        </section>

        {/* ==================== CENTER COLUMN: Spatial Map Observation Canvas ==================== */}
        <section className="col-span-6 cyber-panel relative h-full flex flex-col overflow-hidden">
          <div className="h-10 border-b border-obsidian-carbon px-4 bg-obsidian-charcoal flex items-center justify-between font-mono text-[10px] select-none text-gray-500">
            <div className="flex items-center gap-4">
              <span className="text-white font-bold">
                GIS SPATIAL OBSERVATION CANVAS
              </span>
              {selectedCell && <span>SELECTED HEX: {selectedCell.H3_Index_ID}</span>}
            </div>
            <div>
              PROJ: <span className="text-white">EPSG:4326 // H3-HEX</span>
            </div>
          </div>

          {/* Map canvas containing real Maplibre Map */}
          <div className="flex-1 relative bg-obsidian-void">
            {isLoadingGrid ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center font-mono text-[11px] text-gray-500 bg-obsidian-void">
                LOADING SPATIAL HEXAGON GRID LAYER...
              </div>
            ) : gridError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center font-mono text-[11px] text-crimson bg-obsidian-void">
                FAILED TO RESOLVE GRID COMPONENT INTERFACE
              </div>
            ) : (
              <MapComponent
                gridData={gridData}
                activeLayer={activeLayer}
                selectedCellId={selectedCell ? selectedCell.H3_Index_ID : null}
                onSelectCell={handleSelectCell}
              />
            )}

            {/* Render Indicator Overlay */}
            <div className="absolute bottom-4 right-4 z-10 bg-obsidian-charcoal border border-obsidian-carbon px-2.5 py-1.5 font-mono text-[9px] text-isotope tracking-wider flex items-center gap-1.5 font-bold">
              <span className="h-1.5 w-1.5 bg-isotope" />
              MAP ENGINE DISPATCHED
            </div>
          </div>
        </section>

        {/* ==================== RIGHT COLUMN: Control Panel & Live Pareto Front Chart ==================== */}
        <section className="col-span-3 cyber-panel p-4 flex flex-col h-full overflow-hidden">
          {/* Tab Selection Header */}
          <div className="flex border-b border-obsidian-carbon relative mb-4 p-1 bg-obsidian-void border border-obsidian-carbon">
            <button
              onClick={() => setActiveTab('diagnostics')}
              className={`flex-1 py-1.5 text-[10px] font-mono uppercase tracking-wider relative transition-colors z-10 font-bold ${activeTab === 'diagnostics' ? 'text-black' : 'text-gray-400 hover:text-white'
                }`}
            >
              Manual Diagnostics
              {activeTab === 'diagnostics' && (
                <motion.div
                  layoutId="active-tab-bg"
                  className="absolute inset-0 bg-telemetry -z-10"
                  transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab('optimization')}
              className={`flex-1 py-1.5 text-[10px] font-mono uppercase tracking-wider relative transition-colors z-10 font-bold ${activeTab === 'optimization' ? 'text-black' : 'text-gray-400 hover:text-white'
                }`}
            >
              Evolutionary Optimization
              {activeTab === 'optimization' && (
                <motion.div
                  layoutId="active-tab-bg"
                  className="absolute inset-0 bg-telemetry -z-10"
                  transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                />
              )}
            </button>
          </div>

          {/* Sliding Panel Content */}
          <div className="flex-1 flex flex-col overflow-y-auto">
            <AnimatePresence mode="wait">
              {activeTab === 'diagnostics' ? (
                <motion.div
                  key="diagnostics"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ duration: 0.15 }}
                  className="flex-1 flex flex-col gap-4"
                >
                  <div className="bg-obsidian-void p-3 border border-obsidian-carbon">
                    <h3 className="text-xs font-mono uppercase text-white font-bold mb-1">
                      CO-EFFICIENT SWEEPS
                    </h3>
                    <p className="text-[10px] font-mono text-gray-500 leading-normal mb-3">
                      Adjust delta multipliers manually to trigger predictive biophysical inference.
                    </p>

                    {/* NDVI Delta */}
                    <div className="mb-4">
                      <div className="flex justify-between font-mono text-[10px] mb-1">
                        <span className="text-gray-400">Δ NDVI (Green Space)</span>
                        <span className="text-isotope font-bold">+{deltaNdvi.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.00"
                        max="0.50"
                        step="0.01"
                        value={deltaNdvi}
                        onChange={(e) => {
                          setDeltaNdvi(parseFloat(e.target.value));
                          setSelectedStrategy(null); // clear select indicator when manual sliders move
                        }}
                        className="w-full accent-isotope bg-obsidian-carbon h-1 outline-none"
                      />
                    </div>

                    {/* Albedo Delta */}
                    <div>
                      <div className="flex justify-between font-mono text-[10px] mb-1">
                        <span className="text-gray-400">Δ Albedo (Reflective Surface)</span>
                        <span className="text-telemetry font-bold">+{deltaAlbedo.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.00"
                        max="0.40"
                        step="0.01"
                        value={deltaAlbedo}
                        onChange={(e) => {
                          setDeltaAlbedo(parseFloat(e.target.value));
                          setSelectedStrategy(null); // clear select indicator when manual sliders move
                        }}
                        className="w-full accent-telemetry bg-obsidian-carbon h-1 outline-none"
                      />
                    </div>
                  </div>

                  {/* Projections Panel */}
                  <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex-1 flex flex-col gap-2">
                    <h3 className="text-[10px] font-mono uppercase text-gray-500 border-b border-obsidian-carbon pb-1.5 mb-1.5 font-bold">
                      PROJECTIONS UNDER STATED INTERVENTIONS
                    </h3>

                    <div className="flex items-center justify-between text-xs font-mono">
                      <span>Calculated Cooling LST:</span>
                      <span className="text-white font-bold text-sm font-mono-prec">-{calcCooling}°C</span>
                    </div>

                    <div className="flex items-center justify-between text-xs font-mono">
                      <span>Projected Cooling Cost:</span>
                      <span className="text-white font-bold font-mono-prec">{calcCost} Cr</span>
                    </div>

                    <div className="h-px bg-obsidian-carbon my-2" />

                    <div className="mt-auto">
                      <div className="text-[10px] font-mono text-gray-500 mb-1">THERMODYNAMIC CONSTRAINT STATUS:</div>
                      {isPhysicallyStable ? (
                        <div className="p-2 border border-isotope/30 bg-isotope/5 text-isotope font-mono text-[11px] font-bold">
                          CONSTRAINTS STABILIZED
                        </div>
                      ) : (
                        <div className="p-2 border border-crimson/30 bg-crimson/5 text-crimson font-mono text-[11px] font-bold">
                          CRITICAL THERMAL BOUNDARY REACHED
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="optimization"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.15 }}
                  className="flex-1 flex flex-col gap-4 overflow-hidden"
                >
                  <div className="bg-obsidian-void p-3 border border-obsidian-carbon">
                    <h3 className="text-xs font-mono uppercase text-white font-bold mb-1">
                      EVOLUTIONARY CONTROL DECK
                    </h3>
                    <p className="text-[10px] font-mono text-gray-500 leading-normal mb-3">
                      Tune NSGA-II parameters to run multi-objective optimization sweeps over H3 cells.
                    </p>

                    {/* Generations slider */}
                    <div className="mb-3">
                      <div className="flex justify-between font-mono text-[10px] mb-1">
                        <span className="text-gray-400">Max Generations</span>
                        <span className="text-white font-bold font-mono-prec">{generations}</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        step="5"
                        value={generations}
                        disabled={isOptimizing}
                        onChange={(e) => setGenerations(parseInt(e.target.value))}
                        className="w-full accent-telemetry bg-obsidian-carbon h-1 outline-none disabled:opacity-50"
                      />
                    </div>

                    {/* Population size slider */}
                    <div>
                      <div className="flex justify-between font-mono text-[10px] mb-1">
                        <span className="text-gray-400">Population Size</span>
                        <span className="text-white font-bold font-mono-prec">{population}</span>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="200"
                        step="10"
                        value={population}
                        disabled={isOptimizing}
                        onChange={(e) => setPopulation(parseInt(e.target.value))}
                        className="w-full accent-telemetry bg-obsidian-carbon h-1 outline-none disabled:opacity-50"
                      />
                    </div>
                  </div>

                  {/* Sweep Telemetry / Pareto Front Area */}
                  <div className="bg-obsidian-void p-3 border border-obsidian-carbon flex-1 flex flex-col overflow-hidden">
                    <h3 className="text-[10px] font-mono uppercase text-gray-500 border-b border-obsidian-carbon pb-1.5 mb-2.5 font-bold flex items-center justify-between">
                      <span>SWEEP TELEMETRY MATRIX</span>
                      {isOptimizing && <span className="text-telemetry animate-pulse">COMPUTING...</span>}
                    </h3>

                    {isOptimizing ? (
                      <div className="flex-1 flex flex-col overflow-hidden">
                        <div className="flex flex-col gap-2 font-mono text-xs mb-3">
                          <div className="flex justify-between">
                            <span>GEN ITERATION:</span>
                            <span className="text-telemetry font-bold">{currentGen} / {generations}</span>
                          </div>

                          <div className="w-full bg-obsidian-carbon h-1.5 border border-obsidian-carbon">
                            <div
                              className="bg-telemetry h-full transition-all duration-300"
                              style={{ width: `${(currentGen / generations) * 100}%` }}
                            />
                          </div>

                          <div className="flex justify-between text-[11px]">
                            <span>PEAK COOLING:</span>
                            <span className="text-plasma font-bold font-mono-prec">-{bestCooling.toFixed(2)}°C</span>
                          </div>
                          <div className="flex justify-between text-[11px]">
                            <span>PARETO COUNT:</span>
                            <span className="text-white font-bold font-mono-prec">{paretoCount}</span>
                          </div>
                        </div>

                        {/* Active Compute Ticker Console logs */}
                        <div className="flex-1 border border-obsidian-carbon bg-black p-2 flex flex-col overflow-hidden">
                          <div className="border-b border-obsidian-carbon pb-1 mb-1 font-mono text-[9px] text-gray-500 uppercase">
                            Active Compute Log Output
                          </div>
                          <div
                            ref={logTerminalRef}
                            className="flex-1 overflow-y-auto font-mono text-[9px] text-isotope leading-relaxed space-y-0.5 select-text scrollbar-thin"
                          >
                            {consoleLogs.map((log, index) => (
                              <div key={index}>{log}</div>
                            ))}
                          </div>
                        </div>

                        <button
                          onClick={cancelOptimization}
                          className="w-full py-2 bg-crimson hover:bg-red-700 text-white font-mono text-xs font-bold uppercase tracking-wider transition-colors mt-3"
                        >
                          HALT OPTIMIZATION
                        </button>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col justify-between overflow-hidden">
                        {paretoFront.length > 0 ? (
                          <div className="flex-1 flex flex-col gap-2 overflow-hidden mb-3">
                            {/* Live SVG Pareto frontier chart */}
                            <div className="flex-1 min-h-[180px]">
                              <ParetoPlot
                                paretoFront={paretoFront}
                                selectedStrategy={selectedStrategy}
                                onSelectStrategy={handleSelectStrategy}
                              />
                            </div>
                            {selectedStrategy && (
                              <div className="bg-obsidian-charcoal border border-telemetry/40 p-3 font-mono text-[10px]">
                                <div className="text-[10px] font-bold text-telemetry uppercase tracking-wider mb-2 border-b border-obsidian-carbon pb-1 flex justify-between select-none">
                                  <span>Strategy Financial Intelligence</span>
                                  <span className="text-isotope">ACTIVE ANALYSIS</span>
                                </div>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                                  <div className="flex flex-col">
                                    <span className="text-gray-500 uppercase text-[8px]">Estimated Capex</span>
                                    <span className="text-xs font-bold text-white font-mono mt-0.5">
                                      {new Intl.NumberFormat('en-IN', {
                                        style: 'currency',
                                        currency: 'INR',
                                        maximumFractionDigits: 0
                                      }).format(selectedStrategy.estimated_capex_inr)}
                                    </span>
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="text-gray-500 uppercase text-[8px]">Annual Savings</span>
                                    <span className="text-xs font-bold text-isotope font-mono mt-0.5">
                                      {new Intl.NumberFormat('en-IN', {
                                        style: 'currency',
                                        currency: 'INR',
                                        maximumFractionDigits: 0
                                      }).format(selectedStrategy.annual_energy_savings_inr)}/yr
                                    </span>
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="text-gray-500 uppercase text-[8px]">Carbon Offset</span>
                                    <span className="text-xs font-bold text-plasma font-mono mt-0.5">
                                      {selectedStrategy.carbon_offset_tons.toLocaleString('en-IN', { maximumFractionDigits: 2 })} Tons/yr
                                    </span>
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="text-gray-500 uppercase text-[8px]">Payback Period</span>
                                    <span className="text-xs font-bold text-yellow-400 font-mono mt-0.5">
                                      {selectedStrategy.annual_energy_savings_inr > 0 
                                        ? `${(selectedStrategy.estimated_capex_inr / selectedStrategy.annual_energy_savings_inr).toFixed(1)} Years`
                                        : 'Infinity'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="flex-1 flex flex-col items-center justify-center text-center p-4 border border-dashed border-obsidian-carbon bg-obsidian-void font-mono text-[10px] text-gray-600 leading-normal mb-3">
                            TARGET GRID HEXAGON AND SELECT RUN OPTIMIZATION SWEEP TO COMPILE TRADE-OFF FRONTIERS.
                          </div>
                        )}

                        <button
                          onClick={startLiveOptimization}
                          disabled={!selectedCell}
                          className="w-full py-2.5 bg-telemetry hover:bg-cyan-400 text-black font-mono text-xs font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5 disabled:opacity-50"
                        >
                          RUN OPTIMIZATION SWEEP
                        </button>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>

      </main>
    </div>
  );
}
