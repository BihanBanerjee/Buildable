"""
Base template — locked files that form the foundation of every generated React app.

These files are written to the sandbox by the scaffold node and MUST NOT be modified by the LLM.
The LLM only generates feature files (components, pages, context, utilities).
"""

# All files that form the immutable project skeleton.
# Keys are paths relative to /home/user/react-app/.
BASE_TEMPLATE: dict[str, str] = {
    "package.json": """{
  "name": "buildable-app",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite --host --port 5173",
    "build": "vite build",
    "preview": "vite preview --host --port 5173"
  },
  "dependencies": {
    "react": "^19.1.0",
    "react-dom": "^19.1.0",
    "react-router-dom": "^7.6.1",
    "react-icons": "^5.5.0",
    "lucide-react": "^0.460.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.4.1",
    "vite": "^6.3.5",
    "@tailwindcss/vite": "^4.1.7",
    "tailwindcss": "^4.1.7"
  }
}
""",
    "vite.config.js": """import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: true,
  },
  preview: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: true,
  },
});
""",
    "index.html": """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Buildable App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""",
    "src/main.jsx": """import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",
    "src/index.css": """@import "tailwindcss";
""",
}

# Files that must never be overwritten by the LLM (even via create_file).
LOCKED_FILES = {"vite.config.js", "src/main.jsx", "index.html"}
