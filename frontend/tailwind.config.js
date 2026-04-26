/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          yellow: '#f0c040',
          dark:   '#0d0d0d',
        },
      },
    },
  },
  plugins: [],
}
