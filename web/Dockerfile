FROM node:alpine as builder

WORKDIR /app

COPY package.json /app/package.json

RUN npm install

COPY . /app

RUN npm run build



FROM node:alpine

WORKDIR /app

COPY --from=builder /app/dist /app/dist

COPY package.json /app/package.json

RUN npm install --omit=dev

EXPOSE 4173

USER node

CMD ["npm", "run", "host"]