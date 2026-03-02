import { apiClient } from "./client";
import {
  LoginResponse,
  RegisterResponse,
  LoginRequest,
  RegisterRequest,
  UserData,
} from "./types";

export const authApi = {
  login: async (credentials: LoginRequest): Promise<LoginResponse> => {
    const response = await apiClient.post<LoginResponse>("/auth/login", credentials);
    return response.data;
  },

  register: async (data: RegisterRequest): Promise<RegisterResponse> => {
    const response = await apiClient.post<RegisterResponse>("/auth/register", data);
    return response.data;
  },

  getCurrentUser: async (): Promise<UserData> => {
    const response = await apiClient.get<UserData>("/auth/me");
    return response.data;
  },
};
