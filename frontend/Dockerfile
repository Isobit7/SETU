# =====================================================================
# SETU Frontend — Multi-Stage Dockerfile
# Stage 1: Build React/Vite/TypeScript Production Bundle
# Stage 2: Serve via Nginx Reverse Proxy
# =====================================================================

# Stage 1: Build production bundle
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package manifests from frontend directory
COPY frontend/package.json frontend/package-lock.json* ./

# Install npm dependencies
RUN npm ci || npm install

# Copy frontend source code and configs
COPY frontend/ ./

# Build production bundle using Vite and TypeScript
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Remove default nginx HTML files
RUN rm -rf /usr/share/nginx/html/*

# Copy production build artifacts from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
