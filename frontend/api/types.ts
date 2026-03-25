export interface UserData {
  id: number;
  email: string;
  name: string;
  has_openrouter_key?: boolean;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface RegisterResponse {
  access_token: string;
  token_type: string;
  user: UserData;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export interface ChatResponse {
  status: string;
  message: string;
  chat_id: string;
}

export interface Project {
  id: string;
  user_id: number;
  title: string;
  app_url: string | null;
  deployed_url: string | null;

  created_at: string;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}
