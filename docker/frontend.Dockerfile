FROM node:20-alpine

WORKDIR /app

# Dev server needs vite (@vitejs/plugin-react); ensure devDependencies install even if NODE_ENV=production.
ENV NODE_ENV=development

# Install deps (layer cache)
COPY frontend/package*.json ./
RUN npm install --include=dev

# Copy source
COPY frontend/ ./

EXPOSE 3000

# Bind-mount ./frontend:/app hides image node_modules at runtime; compose runs npm install before dev.
CMD ["sh", "-c", "npm install --include=dev && exec npm run dev -- --host 0.0.0.0 --port 3000"]
