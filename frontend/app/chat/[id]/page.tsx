"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { API_URL } from "@/lib/utils";
import apiClient from "@/api/client";
import {
  ChatIdHeader,
  MessageBubble,
  PreviewPanel,
  ChatInput,
} from "@/components/chat";
import { consolidateMessages } from "@/lib/chat-utils";
import { handleSSEMessage } from "@/lib/sse-handlers";
import type { Message, ActiveToolCall } from "@/lib/chat-types";

export default function ChatIdPage() {
  const params = useParams();
  const router = useRouter();
  const chatId = params.id as string;

  const [sseConnected, setSseConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [appUrl, setAppUrl] = useState<string | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);
  const [previewWidth, setPreviewWidth] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const [userData, setUserData] = useState<any>(null);
  const [currentTool, setCurrentTool] = useState<ActiveToolCall | null>(null);
  const [isCheckingUrl, setIsCheckingUrl] = useState(false);
  const [projectFiles, setProjectFiles] = useState<string[]>([]);
  const [sandboxActive, setSandboxActive] = useState(true);
  const [isRestartingSandbox, setIsRestartingSandbox] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const eventSourceRef = useRef<EventSource | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const urlCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const wasBuilding = useRef(false);

  // Load user data from localStorage
  useEffect(() => {
    const user = localStorage.getItem("user_data");
    if (user) {
      try {
        setUserData(JSON.parse(user));
      } catch (err) {
        console.error("Failed to parse user data:", err);
      }
    }
    setIsLoading(false);
  }, []);

  // Fetch project files from backend
  const fetchProjectFiles = async () => {
    if (typeof window === "undefined") return;
    try {
      const token = localStorage.getItem("auth_token");
      if (!token) return;

      const response = await apiClient.get<{
        project_id: string;
        files: string[];
        sandbox_id: string;
        sandbox_active: boolean;
      }>(`/projects/${chatId}/files`);

      setProjectFiles(response.data.files || []);
      setSandboxActive(response.data.sandbox_active ?? true);
    } catch (error) {
      console.error("Error fetching files:", error);
    }
  };

  const restartSandbox = async () => {
    setIsRestartingSandbox(true);
    try {
      const response = await apiClient.post<{ app_url: string }>(
        `/projects/${chatId}/restart`
      );
      setAppUrl(response.data.app_url);
      setSandboxActive(true);
    } catch (err) {
      console.error("Failed to restart sandbox:", err);
    } finally {
      setIsRestartingSandbox(false);
    }
  };

  // Check if the app URL is responding
  const checkUrlReady = async (url: string): Promise<boolean> => {
    try {
      await fetch(url, { method: "HEAD", mode: "no-cors" });
      return true;
    } catch {
      return false;
    }
  };

  // Poll until the preview URL is available, then set it
  const pollUrlUntilReady = async (url: string) => {
    setIsCheckingUrl(true);
    let attempts = 0;
    const maxAttempts = 20;

    const checkInterval = setInterval(async () => {
      attempts++;
      const isReady = await checkUrlReady(url);

      if (isReady || attempts >= maxAttempts) {
        clearInterval(checkInterval);
        setIsCheckingUrl(false);
        setAppUrl(url);
      }
    }, 1000);

    urlCheckIntervalRef.current = checkInterval;
  };

  // Cleanup poll interval on unmount
  useEffect(() => {
    return () => {
      if (urlCheckIntervalRef.current) clearInterval(urlCheckIntervalRef.current);
    };
  }, []);

  // Fetch files when appUrl becomes available (handles page reload via history event)
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!appUrl || !chatId) return;

    const t = setTimeout(() => fetchProjectFiles(), 1000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appUrl, chatId]);

  // Fetch files when build transitions from in-progress → done
  // (wasBuilding ref prevents firing on every render)
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!chatId) return;

    if (isBuilding) {
      wasBuilding.current = true;
      return;
    }

    if (wasBuilding.current) {
      wasBuilding.current = false;
      const t = setTimeout(() => fetchProjectFiles(), 2000);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBuilding, chatId]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Drag-to-resize the chat/preview split
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const chatWidth = ((e.clientX - rect.left) / rect.width) * 100;
      const newPreviewWidth = 100 - chatWidth;
      if (chatWidth > 20 && chatWidth < 70) setPreviewWidth(newPreviewWidth);
    };

    const handleMouseUp = () => setIsDragging(false);

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging]);

  // SSE connection — opens once per chatId, cleaned up on unmount
  useEffect(() => {
    if (eventSourceRef.current) return;

    const token = localStorage.getItem("auth_token");
    if (!token) return;

    const timer = setTimeout(() => {
      try {
        const sseUrl = `${API_URL}/sse/${chatId}?token=${token}`;
        const es = new EventSource(sseUrl);

        es.onopen = () => {
          setSseConnected(true);
          setError(null);
        };

        es.onerror = () => {
          setSseConnected(false);
          // EventSource auto-reconnects
        };

        es.onmessage = (event) => {
          handleSSEMessage(event, {
            setCurrentTool,
            setIsBuilding,
            setIsSending,
            pollUrlUntilReady,
            setMessages,
            setAppUrl,
            setError,
            setUserData,
            consolidateMessages,
            currentTool,
          });
        };

        eventSourceRef.current = es;
      } catch (err) {
        console.error("SSE connection failed:", err);
        setSseConnected(false);
      }
    }, 100);

    return () => {
      clearTimeout(timer);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [chatId]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !sseConnected || isBuilding || isSending) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const promptText = input.trim();
    setInput("");
    // Don't set isBuilding here — let SSE events control it.
    // The "started" event sets isBuilding=true for real builds,
    // while "chat_response" events skip it entirely. This prevents
    // the preview panel from flashing briefly on chat-classified prompts.
    setIsSending(true);

    try {
      await apiClient.post(`/chats/${chatId}/messages`, { prompt: promptText });
    } catch (err) {
      console.error("Failed to send message:", err);
      setError("Failed to send message. Please try again.");
      setIsSending(false);
    }
  };

  // Auto-hide preview panel when there's nothing to show (e.g. chat-only responses)
  const hasPreviewContent = !!(appUrl || isBuilding || projectFiles.length > 0);
  const shouldShowPreview = showPreview && hasPreviewContent;

  return (
    <div
      className="min-h-screen w-full relative overflow-hidden"
      style={{ backgroundColor: "#030712" }}
      ref={containerRef}
    >
      {/* Emerald glow — top-left corner, subtle */}
      <div
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 50% 80% at 0% 0%, rgba(16,185,129,0.08), transparent 60%)",
        }}
      />

      <div className="relative z-10 h-screen flex flex-col">
        <ChatIdHeader
          userData={userData}
          showPreview={showPreview}
          onTogglePreview={() => setShowPreview(!showPreview)}
          onNewChat={() => router.push("/chat")}
          onBack={() => router.push("/chat")}
          onUserDataUpdate={setUserData}
        />

        {/* Main split layout */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat panel */}
          <div
            className="flex flex-col border-r border-border"
            style={{
              width: shouldShowPreview ? `${100 - previewWidth}%` : "100%",
              transition: isDragging ? "none" : "width 0.25s ease-out",
            }}
          >
            {/* Message list */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span className="text-sm">Loading...</span>
                  </div>
                </div>
              ) : error ? (
                <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg text-sm">
                  {error}
                </div>
              ) : messages.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center text-muted-foreground/50">
                    <div className="w-10 h-10 rounded-full border border-border flex items-center justify-center mx-auto mb-3">
                      <span className="text-primary text-lg font-bold">B</span>
                    </div>
                    <p className="text-sm">Build is starting…</p>
                  </div>
                </div>
              ) : null}

              {messages.map((msg, index) => (
                <MessageBubble
                  key={index}
                  message={msg}
                  isLastMessage={index === messages.length - 1}
                />
              ))}

              <div ref={messagesEndRef} />
            </div>

            <ChatInput
              input={input}
              wsConnected={sseConnected}
              isBuilding={isBuilding || isSending}
              onInputChange={setInput}
              onSubmit={handleSendMessage}
            />
          </div>

          {/* Drag divider */}
          {shouldShowPreview && (
            <div
              className="w-1 bg-border hover:bg-primary/30 cursor-col-resize transition-colors duration-150"
              onMouseDown={() => setIsDragging(true)}
              style={{ userSelect: "none" }}
            />
          )}

          {/* Preview panel */}
          {shouldShowPreview && (
            <PreviewPanel
              appUrl={appUrl}
              isCheckingUrl={isCheckingUrl}
              previewWidth={previewWidth}
              files={projectFiles}
              projectId={chatId}
              sandboxActive={sandboxActive}
              isRestartingSandbox={isRestartingSandbox}
              onRestartSandbox={restartSandbox}
            />
          )}
        </div>
      </div>
    </div>
  );
}
