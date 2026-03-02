import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// Inject auth token on every request
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("auth_token");
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Global 401 handler — clear storage, redirect to sign-in
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("auth_token");
      localStorage.removeItem("user_data");
      if (!window.location.pathname.includes("/signin")) {
        window.location.href = "/signin";
      }
    }
    const errorMessage =
      (error.response?.data as { detail?: string })?.detail || error.message;
    return Promise.reject(new Error(errorMessage));
  },
);

export default apiClient;
