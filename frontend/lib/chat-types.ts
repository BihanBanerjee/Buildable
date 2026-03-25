export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  formatted?: string;
  created_at: string;
  event_type?: string;
  isCompleted?: boolean;
  isSuccess?: boolean;
  isProgress?: boolean;
  summary?: string;
  buildDuration?: number;
  buildFiles?: string[];
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
  [key: string]: unknown;
}

export type BuildStage = "building" | "validating" | "completed";

export interface FileActivity {
  path: string;
  action: string;
  timestamp: number;
}

export interface SSEHandlers {
  setCurrentTool: (tool: ActiveToolCall | null) => void;
  setIsBuilding: (isBuilding: boolean) => void;
  setIsSending: (isSending: boolean) => void;
  setBuildStage: (stage: BuildStage | null) => void;
  pollUrlUntilReady: (url: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setAppUrl: (url: string | null) => void;
  setError: (error: string | null) => void;
  setUserData: (data: any) => void;
  consolidateMessages: (messages: Message[]) => Message[];
  setFileActivities?: React.Dispatch<React.SetStateAction<FileActivity[]>>;
  setBuildLogs?: React.Dispatch<React.SetStateAction<string[]>>;
  setDeployedUrl?: (url: string | null) => void;
}
