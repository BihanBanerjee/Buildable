export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  formatted?: string;
  created_at: string;
  event_type?: string;
  isCompleted?: boolean;
  summary?: string;
  tool_calls?: Array<{
    name: string;
    status: "success" | "error" | "running";
    output?: string;
    detail?: string;
    input?: string;
  }>;
}

export interface ActiveToolCall {
  name: string;
  status: "running" | "completed";
  output?: string;
}

export interface SSEMessage {
  type?: string;
  e?: string;
  message?: string;
  content?: any;
  formatted?: string;
  url?: string;
  app_url?: string;
  messages?: Message[];
  tool_name?: string;
  tool_input?: Record<string, any>;
  tool_output?: string | object;
  tokens_remaining?: number;
  reset_in_hours?: number;
  [key: string]: unknown;
}

export interface SSEHandlers {
  setCurrentTool: (tool: ActiveToolCall | null) => void;
  setIsBuilding: (isBuilding: boolean) => void;
  pollUrlUntilReady: (url: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setAppUrl: (url: string | null) => void;
  setError: (error: string | null) => void;
  setUserData: (data: any) => void;
  consolidateMessages: (messages: Message[]) => Message[];
  currentTool: ActiveToolCall | null;
}
