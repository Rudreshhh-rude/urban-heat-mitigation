import React, { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const API_URL = import.meta.env.VITE_API_BASE_URL;
const MAP_STYLE_URL = import.meta.env.VITE_MAP_STYLE_URL || 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const MapComponent = React.memo(function MapComponent({ gridData, activeLayer, selectedCellId, onSelectCell }) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Default center for Bengaluru coordinates
    const defaultCenter = [77.62, 12.97];

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE_URL,
      center: defaultCenter,
      zoom: 11,
      minZoom: 9,
      maxZoom: 16,
      attributionControl: false
    });

    mapRef.current = map;

    map.on('load', () => {
      console.log('[MAP] Maplibre canvas loaded.');

      // Add grid source if gridData is present
      if (gridData && gridData.features && gridData.features.length > 0) {
        initializeGridSource(map, gridData);
        updateMapLayers(map, activeLayer);
        updateSelectedHighlight(map, selectedCellId);
      }
    });

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Update source data when gridData changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const source = map.getSource('h3-grid');
    if (source) {
      source.setData(gridData);
    } else if (gridData && gridData.features && gridData.features.length > 0) {
      initializeGridSource(map, gridData);
      updateMapLayers(map, activeLayer);
    }
  }, [gridData]);

  // Update active layer visualization
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded() || !map.getSource('h3-grid')) return;

    updateMapLayers(map, activeLayer);
  }, [activeLayer]);

  // Update highlighted hexagon selection
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded() || !map.getSource('h3-grid')) return;

    updateSelectedHighlight(map, selectedCellId);
  }, [selectedCellId]);

  const initializeGridSource = (map, data) => {
    // Prevent duplicate sources
    if (map.getSource('h3-grid')) return;

    map.addSource('h3-grid', {
      type: 'geojson',
      data: data
    });

    // 1. Grid Fill Layer
    map.addLayer({
      id: 'h3-grid-fill',
      type: 'fill',
      source: 'h3-grid',
      paint: {
        'fill-opacity': 0.75,
        'fill-color': '#16161f'
      }
    });

    // 2. Base Grid Borders (thin high-contrast lines)
    map.addLayer({
      id: 'h3-grid-borders',
      type: 'line',
      source: 'h3-grid',
      paint: {
        'line-color': '#27272a',
        'line-width': 0.75,
        'line-opacity': 0.9
      }
    });

    // 3. Selection Highlight Layer
    map.addLayer({
      id: 'h3-grid-selected',
      type: 'line',
      source: 'h3-grid',
      paint: {
        'line-color': '#d97706',
        'line-width': 2.5,
        'line-opacity': 1.0
      },
      filter: ['==', ['get', 'H3_Index_ID'], '']
    });

    // Setup map interactions
    map.on('click', 'h3-grid-fill', (e) => {
      if (e.features && e.features.length > 0) {
        const props = e.features[0].properties;
        console.log('[MAP] Selected cell properties:', props);
        onSelectCell(props);
      }
    });

    map.on('mouseenter', 'h3-grid-fill', () => {
      map.getCanvas().style.cursor = 'pointer';
    });

    map.on('mouseleave', 'h3-grid-fill', () => {
      map.getCanvas().style.cursor = '';
    });
  };

  const updateMapLayers = (map, layerId) => {
    if (!map.getLayer('h3-grid-fill')) return;

    let fillColorExpression;

    switch (layerId) {
      case 'lst': // Land Surface Temperature Ramp (Cool to Hot: Obsidian -> Telemetry -> Plasma -> Crimson)
        fillColorExpression = [
          'interpolate',
          ['linear'],
          ['get', 'LST'],
          30.0, '#16161f',
          42.0, '#00f0ff',
          48.0, '#ff5a00',
          55.0, '#de0a26'
        ];
        break;
      case 'ndvi': // Vegetation Fraction Ramp (Low to High: Charcoal -> Telemetry -> Isotope Green)
        fillColorExpression = [
          'interpolate',
          ['linear'],
          ['get', 'NDVI'],
          -0.1, '#0e0e12',
          0.1, '#16161f',
          0.3, '#00f0ff',
          0.6, '#39ff14'
        ];
        break;
      case 'albedo': // Surface Reflectivity Ramp (Low to High: Void -> Telemetry -> Pure White)
        fillColorExpression = [
          'interpolate',
          ['linear'],
          ['get', 'Albedo'],
          0.1, '#050507',
          0.2, '#16161f',
          0.35, '#00f0ff',
          0.6, '#ffffff'
        ];
        break;
      case 'building': // Building Density Ramp
        fillColorExpression = [
          'interpolate',
          ['linear'],
          ['get', 'Building_Density'],
          0.0, '#0e0e12',
          0.1, '#16161f',
          0.4, '#00f0ff',
          0.8, '#ffffff'
        ];
        break;
      default:
        fillColorExpression = '#16161f';
    }

    map.setPaintProperty('h3-grid-fill', 'fill-color', fillColorExpression);
  };

  const updateSelectedHighlight = (map, selectedId) => {
    if (!map.getLayer('h3-grid-selected')) return;

    map.setFilter('h3-grid-selected', [
      '==',
      ['get', 'H3_Index_ID'],
      selectedId || ''
    ]);
  };

  return (
    <div className="absolute inset-0 w-full h-full bg-obsidian-void">
      <div ref={mapContainerRef} className="w-full h-full" />
    </div>
  );
});

export default MapComponent;
