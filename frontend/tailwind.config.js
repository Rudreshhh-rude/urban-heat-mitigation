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
          void: '#050507',
          charcoal: '#0e0e12',
          carbon: '#16161f'
        },
        telemetry: '#00f0ff',
        plasma: '#ff5a00',
        crimson: '#de0a26',
        isotope: '#39ff14'
      },
      fontFamily: {
        sans: ['Geist', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      }
    },
  },
  plugins: [],
}
