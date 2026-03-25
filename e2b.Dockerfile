FROM node:20-slim

# Set up user directory (matches E2B default)
RUN mkdir -p /home/user/react-app && chown -R 1000:1000 /home/user

WORKDIR /home/user/react-app

# Copy package.json and install deps so node_modules is pre-baked
COPY e2b-package.json ./package.json
RUN npm install

# Ensure correct ownership
RUN chown -R 1000:1000 /home/user/react-app
