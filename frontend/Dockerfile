# Stage 1: Build Vite production assets
FROM node:20-alpine AS build

WORKDIR /app

# Copy package configurations
COPY package*.json ./

# Install packages
RUN npm ci

# Copy source code files
COPY . .

# Set Vite API base path to relative /api for unified Docker proxying
ENV VITE_API_BASE_URL=/api

# Compile production bundle
RUN npm run build

# Stage 2: Serve static bundle via Nginx
FROM nginx:alpine

# Remove default server config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom server config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled assets from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Expose Vite port (5173) mapped to Nginx inside the container
EXPOSE 5173

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
