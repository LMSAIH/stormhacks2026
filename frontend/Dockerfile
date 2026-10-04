FROM node:22-alpine AS build

WORKDIR /app

RUN corepack enable

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile --config.strict-dep-builds=false

COPY . .

ARG VITE_API_URL=http://localhost:4000
ARG VITE_TTS_WS_URL=ws://localhost:8765
ARG VITE_LIPREAD_URL=
ARG VITE_LIPREAD_MODEL_BASE=
ARG VITE_ORT_WEBGPU=0

ENV VITE_API_URL=${VITE_API_URL} \
    VITE_TTS_WS_URL=${VITE_TTS_WS_URL} \
    VITE_LIPREAD_URL=${VITE_LIPREAD_URL} \
    VITE_LIPREAD_MODEL_BASE=${VITE_LIPREAD_MODEL_BASE} \
    VITE_ORT_WEBGPU=${VITE_ORT_WEBGPU}

RUN pnpm build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80