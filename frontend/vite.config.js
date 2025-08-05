import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import crypto from 'crypto'

globalThis.crypto ??= {}
globalThis.crypto.subtle ??= {
  digest: async (algorithm, data) => {
    return crypto.createHash(algorithm.toLowerCase().replace('-', '')).update(Buffer.from(data)).digest()
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    }
  }
})
