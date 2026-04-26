import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // API routes
      '/users':         { target: 'http://localhost:9000', changeOrigin: true },
      '/violations':    { target: 'http://localhost:9000', changeOrigin: true },
      '/payments':      { target: 'http://localhost:9000', changeOrigin: true },
      '/detections':    { target: 'http://localhost:9000', changeOrigin: true },
      '/source_status': { target: 'http://localhost:9000', changeOrigin: true },
      // Video / camera controls
      '/upload':        { target: 'http://localhost:9000', changeOrigin: true },
      '/use_webcam':    { target: 'http://localhost:9000', changeOrigin: true },
      // MJPEG stream — must disable compression and buffering
      '/video': {
        target:      'http://localhost:9000',
        changeOrigin: true,
        ws:           false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.removeHeader('accept-encoding')
          })
        },
      },
    },
  },
})
