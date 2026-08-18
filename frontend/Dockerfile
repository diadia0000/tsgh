# syntax=docker/dockerfile:1

# ============================================================
# Dockerfile for tsgh frontend — Vite dev server (React + TS)
# ============================================================

FROM node:26-slim

WORKDIR /app

# 只複製 lock 相關檔案，讓 npm ci 這層可被 cache（原始碼變動不會使其失效）
COPY package.json package-lock.json ./
RUN npm ci

# 注意：故意不 COPY 原始碼進 image。docker-compose.yml 會用 bind mount
# 把 frontend/ 整個蓋掉 /app，COPY 進去的內容永遠不會被用到（同 backend
# Dockerfile 的教訓，見 git log）。node_modules 靠上面 npm ci 留在 image
# 裡，compose 那邊要用 anonymous volume（- /app/node_modules）蓋住 bind
# mount，不然 host 沒裝過 node_modules 的話 mount 進去會是空的。

EXPOSE 5173

CMD ["npm", "run", "dev"]
