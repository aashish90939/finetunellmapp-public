FROM node:20-alpine AS build

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json ./
RUN npm install

# Copy source and build
COPY frontend ./
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=build /app/frontend/dist ./dist
RUN npm install -g serve
EXPOSE 4173
CMD ["serve", "-s", "dist", "-l", "4173"]
