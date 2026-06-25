/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: {
          void: '#09090b',       // Zinc Jet (background)
          charcoal: '#18181b',   // Zinc Oxide (secondary panels)
          carbon: '#27272a'      // Muted Zinc Gridlines/Borders
        },
        telemetry: '#d97706',    // High-Vis Industrial Amber (Chroma)
        plasma: '#71717a',       // Muted Zinc
        crimson: '#71717a',      // Muted Zinc
        isotope: '#d97706'       // High-Vis Industrial Amber (Chroma)
      },
      fontFamily: {
        sans: ['Roboto', 'sans-serif'],
        mono: ['Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
