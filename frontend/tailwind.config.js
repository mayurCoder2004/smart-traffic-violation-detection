/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    screens: {
      xs:  '375px',
      sm:  '640px',
      md:  '768px',
      lg:  '1024px',
      xl:  '1280px',
      '2xl': '1536px',
    },
    extend: {
      colors: {
        brand: {
          yellow: '#f0c040',
          dark:   '#050508',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'shimmer':    'shimmer 2.5s linear infinite',
        'float':      'float 4s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 2.2s ease-out infinite',
        'scan':       'scan 2.4s linear infinite',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
      },
      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '-200% center' },
          '100%': { backgroundPosition:  '200% center' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':       { transform: 'translateY(-8px)' },
        },
        'pulse-ring': {
          '0%':   { boxShadow: '0 0 0 0 rgba(240,192,64,0.45)' },
          '70%':  { boxShadow: '0 0 0 10px rgba(240,192,64,0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(240,192,64,0)' },
        },
        scan: {
          '0%':   { top: '-4px', opacity: '0' },
          '5%':   { opacity: '1' },
          '95%':  { opacity: '1' },
          '100%': { top: '100%', opacity: '0' },
        },
        'glow-pulse': {
          '0%, 100%': { boxShadow: '0 0 15px rgba(240,192,64,0.15)' },
          '50%':       { boxShadow: '0 0 35px rgba(240,192,64,0.35)' },
        },
      },
    },
  },
  plugins: [],
}
