import basicSsl from '@vitejs/plugin-basic-ssl';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * `npm run dev`        normal development, http://localhost:5173
 * `npm run dev:phone`  same, but served over HTTPS on your Wi-Fi network so a
 *                      phone can open it and use its camera.
 *
 * Why HTTPS: browsers only allow camera access on secure pages (https) or on
 * localhost. A phone opening http://192.168.x.x:5173 is neither, so the
 * camera button would be refused. The certificate is self-signed — the phone
 * shows a warning the first time; choose "Advanced" → "Proceed".
 */
export default defineConfig(({ mode }) => {
  const phone = mode === 'phone';
  return {
    plugins: [react(), ...(phone ? [basicSsl()] : [])],
    // Relative base so the built dashboard also works when served from a
    // subfolder or opened from a static host, not just from the domain root.
    base: './',
    server: {
      port: 5173,
      proxy: { '/api': 'http://127.0.0.1:8000', '/uploads': 'http://127.0.0.1:8000' },
      // dev:phone listens on the network, not just this computer.
      host: phone ? true : undefined,
      open: false,
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
