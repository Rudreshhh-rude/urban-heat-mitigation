import React, { useState } from 'react';

const ParetoPlot = React.memo(function ParetoPlot({ paretoFront, selectedStrategy, onSelectStrategy }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  if (!paretoFront || paretoFront.length === 0) {
    return (
      <div className="h-full flex items-center justify-center border border-dashed border-obsidian-carbon bg-obsidian-void p-4 text-center">
        <span className="text-[10px] font-mono text-gray-600 uppercase">
          NO ACTIVE SEARCH SWEEP COMPILED
        </span>
      </div>
    );
  }

  // Dimensions
  const width = 360;
  const height = 220;
  const margin = { top: 15, right: 15, bottom: 35, left: 45 };

  // Calculate dynamic boundaries
  const maxCost = Math.max(...paretoFront.map(d => d.cost), 1.0);
  const maxLstDrop = Math.max(...paretoFront.map(d => d.lst_drop), 5.0);

  // Buffer scales
  const xMax = maxCost * 1.15;
  const yMax = maxLstDrop * 1.15;

  const getX = (val) => {
    return margin.left + (val / xMax) * (width - margin.left - margin.right);
  };

  const getY = (val) => {
    return height - margin.bottom - (val / yMax) * (height - margin.top - margin.bottom);
  };

  // Sort Pareto Front by cost (X-axis) to draw the trade-off frontier curve
  const sortedFront = [...paretoFront].sort((a, b) => a.cost - b.cost);

  // Generate frontier line path D attribute
  let linePathD = '';
  if (sortedFront.length > 1) {
    linePathD = sortedFront.reduce((path, pt, idx) => {
      const cx = getX(pt.cost);
      const cy = getY(pt.lst_drop);
      return idx === 0 ? `M ${cx} ${cy}` : `${path} L ${cx} ${cy}`;
    }, '');
  }

  // Define ticks
  const xTicks = [0, maxCost * 0.25, maxCost * 0.5, maxCost * 0.75, maxCost].map(v => Number(v.toFixed(2)));
  const yTicks = [0, maxLstDrop * 0.25, maxLstDrop * 0.5, maxLstDrop * 0.75, maxLstDrop].map(v => Number(v.toFixed(2)));

  return (
    <div className="relative flex flex-col w-full bg-obsidian-void p-2 border border-obsidian-carbon">
      <div className="flex justify-between items-center mb-1 select-none">
        <span className="text-[9px] font-mono uppercase tracking-wider text-gray-500 font-bold">
          Pareto Trade-Off Frontier
        </span>
        <span className="text-[9px] font-mono text-telemetry">
          OPTIMAL OPTIONS: {paretoFront.length}
        </span>
      </div>

      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
        {/* Background Grids */}
        {xTicks.map((val, idx) => (
          <line
            key={`grid-x-${idx}`}
            x1={getX(val)}
            y1={margin.top}
            x2={getX(val)}
            y2={height - margin.bottom}
            stroke="#27272a"
            strokeWidth="0.5"
          />
        ))}
        {yTicks.map((val, idx) => (
          <line
            key={`grid-y-${idx}`}
            x1={margin.left}
            y1={getY(val)}
            x2={width - margin.right}
            y2={getY(val)}
            stroke="#27272a"
            strokeWidth="0.5"
          />
        ))}

        {/* X & Y Axes */}
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
          stroke="#27272a"
          strokeWidth="1"
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="#27272a"
          strokeWidth="1"
        />

        {/* X Axis Ticks */}
        {xTicks.map((val, idx) => (
          <g key={`tick-x-${idx}`}>
            <line
              x1={getX(val)}
              y1={height - margin.bottom}
              x2={getX(val)}
              y2={height - margin.bottom + 4}
              stroke="#27272a"
              strokeWidth="1"
            />
            <text
              x={getX(val)}
              y={height - margin.bottom + 15}
              fill="#71717a"
              fontSize="8"
              className="font-mono"
              textAnchor="middle"
            >
              {val}
            </text>
          </g>
        ))}

        {/* Y Axis Ticks */}
        {yTicks.map((val, idx) => (
          <g key={`tick-y-${idx}`}>
            <line
              x1={margin.left - 4}
              y1={getY(val)}
              x2={margin.left}
              y2={getY(val)}
              stroke="#27272a"
              strokeWidth="1"
            />
            <text
              x={margin.left - 8}
              y={getY(val) + 3}
              fill="#71717a"
              fontSize="8"
              className="font-mono"
              textAnchor="end"
            >
              {val}
            </text>
          </g>
        ))}

        {/* Axis Labels */}
        <text
          x={margin.left + (width - margin.left - margin.right) / 2}
          y={height - 5}
          fill="#f4f4f5"
          fontSize="9"
          className="font-mono"
          textAnchor="middle"
          fontWeight="bold"
        >
          INTERVENTION COST (CR)
        </text>

        <text
          x={10}
          y={margin.top + (height - margin.top - margin.bottom) / 2}
          fill="#f4f4f5"
          fontSize="9"
          className="font-mono"
          textAnchor="middle"
          fontWeight="bold"
          transform={`rotate(-90 10 ${margin.top + (height - margin.top - margin.bottom) / 2})`}
        >
          LST TEMPERATURE DROP (°C)
        </text>

        {/* Frontier Curve Path */}
        {linePathD && (
          <path
            d={linePathD}
            fill="none"
            stroke="#27272a"
            strokeWidth="1.5"
          />
        )}
        {linePathD && (
          <path
            d={linePathD}
            fill="none"
            stroke="#d97706"
            strokeWidth="1"
            strokeDasharray="2,3"
            opacity="0.7"
          />
        )}

        {/* Data points */}
        {sortedFront.map((pt, idx) => {
          const cx = getX(pt.cost);
          const cy = getY(pt.lst_drop);
          
          const isSelected = selectedStrategy && 
            Math.abs(selectedStrategy.cost - pt.cost) < 1e-4 && 
            Math.abs(selectedStrategy.lst_drop - pt.lst_drop) < 1e-4;

          const isHovered = hoveredPoint && 
            Math.abs(hoveredPoint.cost - pt.cost) < 1e-4 && 
            Math.abs(hoveredPoint.lst_drop - pt.lst_drop) < 1e-4;

          return (
            <g key={`pt-${idx}`} className="cursor-pointer">
              {/* Highlight Ring for Selected/Hovered */}
              {(isSelected || isHovered) && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 7 : 6}
                  fill="none"
                  stroke="#d97706"
                  strokeWidth="1.5"
                  opacity={isSelected ? 1.0 : 0.6}
                />
              )}

              {/* Point Node */}
              <circle
                cx={cx}
                cy={cy}
                r="3.5"
                fill={isSelected ? '#d97706' : '#71717a'}
                stroke={isHovered ? '#f4f4f5' : '#18181b'}
                strokeWidth="1"
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => onSelectStrategy(pt)}
              />
            </g>
          );
        })}
      </svg>

      {/* Numerical Data Readout overlay inside chart panel */}
      <div className="h-12 border-t border-obsidian-carbon mt-2 pt-1.5 flex justify-between font-mono text-[10px]">
        {hoveredPoint || selectedStrategy ? (
          (() => {
            const data = hoveredPoint || selectedStrategy;
            const isHover = !!hoveredPoint;
            return (
              <div className="flex justify-between w-full">
                <div>
                  <span className="text-gray-500 uppercase">{isHover ? 'Hover' : 'Select'}:</span>{' '}
                  <span className="text-white font-bold font-mono">-{data.lst_drop.toFixed(2)}°C</span>
                </div>
                <div>
                  <span className="text-gray-500">CAPEX:</span>{' '}
                  <span className="text-white font-bold font-mono">
                    {data.estimated_capex_inr !== undefined
                      ? `₹${(data.estimated_capex_inr / 1e7).toFixed(2)} Cr`
                      : `${data.cost.toFixed(2)} Cr`}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">PAYBACK:</span>{' '}
                  <span className="text-yellow-400 font-bold font-mono">
                    {data.annual_energy_savings_inr && data.annual_energy_savings_inr > 0
                      ? `${(data.estimated_capex_inr / data.annual_energy_savings_inr).toFixed(1)} Yr`
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="text-isotope font-mono">ΔN: +{data.delta_ndvi.toFixed(2)}</span>
                  <span className="text-telemetry font-mono">ΔA: +{data.delta_albedo.toFixed(2)}</span>
                </div>
              </div>
            );
          })()
        ) : (
          <div className="text-gray-500 uppercase flex items-center justify-center w-full text-center">
            Hover or click nodes to select cooling strategy
          </div>
        )}
      </div>
    </div>
  );
});

export default ParetoPlot;
