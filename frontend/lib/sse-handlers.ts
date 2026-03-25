import type { SSEMessage, SSEHandlers } from "./chat-types";

export function handleSSEMessage(event: MessageEvent, handlers: SSEHandlers) {
  try {
    const data: SSEMessage = JSON.parse(event.data);
    console.log("SSE:", data.e || data.type);

    // ── log events → accumulate in build logs for terminal display ──
    if (data.e === "log") {
      const logMessage = (data.message as string) || "";
      if (!logMessage) return;
      handlers.setBuildLogs?.((prev) => [...prev, logMessage]);
      return;
    }

    // ── token events → streamed LLM text ──
    if (data.e === "token") {
      const content = (data.content as string) || "";
      if (!content) return;
      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant" && !lastMsg.isProgress) {
          return [...prev.slice(0, -1), { ...lastMsg, content: (lastMsg.content || "") + content }];
        }
        return [...prev, {
          id: Date.now().toString() + "-token",
          role: "assistant" as const,
          content: content,
          created_at: new Date().toISOString(),
          event_type: "token",
        }];
      });
      return;
    }

    // ── file_update events → file activity tracking ──
    if (data.e === "file_update") {
      const file = data.file as { path: string; action: string } | undefined;
      if (!file?.path) return;
      handlers.setFileActivities?.((prev) => [
        ...prev,
        { path: file.path, action: file.action || "modify", timestamp: Date.now() },
      ]);
      return;
    }

    // ── status events → validation progress ──
    if (data.e === "status") {
      const statusMsg = (data.message as string) || "";
      if (!statusMsg) return;
      handlers.setBuildStage?.("validating");
      handlers.setBuildLogs?.((prev) => [...prev, statusMsg]);
      return;
    }

    // ── warning events ──
    if (data.e === "warning") {
      const warnMsg = (data.message as string) || "";
      if (!warnMsg) return;
      handlers.setMessages((prev) => [...prev, {
        id: Date.now().toString() + "-warn",
        role: "assistant" as const,
        content: `⚠️ ${warnMsg}`,
        created_at: new Date().toISOString(),
        event_type: "warning",
      }]);
      return;
    }

    // ── started → begin build ──
    if (data.e === "started") {
      handlers.setIsSending(false);
      handlers.setIsBuilding(true);
      handlers.setBuildStage("building");
      handlers.setBuildLogs?.([]); // Reset logs for new build
    }

    if (data.url) {
      handlers.setIsBuilding(false);
      handlers.pollUrlUntilReady(data.url);
    }

    // ── cancelled ──
    if (data.e === "cancelled") {
      handlers.setIsBuilding(false);
      handlers.setCurrentTool(null);
      handlers.setBuildStage(null);
      handlers.setBuildLogs?.([]);
      handlers.setMessages((prev) => {
        return prev.map((msg, idx) => {
          if (idx === prev.length - 1 && msg.role === "assistant") {
            return { ...msg, content: msg.content || "Build cancelled." };
          }
          return msg;
        });
      });
      return;
    }

    // ── completed ──
    if (data.e === "completed") {
      handlers.setIsBuilding(false);
      handlers.setCurrentTool(null);
      handlers.setBuildStage("completed");
      handlers.setBuildLogs?.([]);
      handlers.setIsSending(false);

      // Show summary for actual builds (which have duration_s), not chat responses
      const wasBuild = data.duration_s != null || data.files != null;
      if (wasBuild) {
        const duration = (data.duration_s as number) || undefined;
        const isSuccess = data.success !== false;
        const buildFiles = (data.files as string[]) || undefined;

        if (!isSuccess) {
          const errorMsg = "Build completed with errors. Some features may not work correctly.";
          handlers.setError(errorMsg);
        }

        handlers.setMessages((prev) => {
          const cleaned = prev.filter((msg) => !msg.isProgress);

          // Mark the last assistant message as completed
          const lastAssistantIdx = cleaned.findLastIndex(
            (msg) => msg.role === "assistant"
          );

          const completedFields = {
            isCompleted: true,
            isSuccess,
            isProgress: false,
            ...(duration ? { buildDuration: duration } : {}),
            ...(buildFiles ? { buildFiles } : {}),
          };

          if (lastAssistantIdx >= 0) {
            return cleaned.map((msg, idx) => {
              if (idx === lastAssistantIdx) {
                return { ...msg, ...completedFields };
              }
              return msg;
            });
          }

          // No assistant message exists — create one
          return [...cleaned, {
            id: Date.now().toString() + "-done",
            role: "assistant" as const,
            content: "Build complete.",
            created_at: new Date().toISOString(),
            event_type: "completed",
            ...completedFields,
          }];
        });
      }

      const updatedUser = localStorage.getItem("user_data");
      if (updatedUser) {
        handlers.setUserData(JSON.parse(updatedUser));
      }
    }

    // ── history ──
    if (data.e === "history" && data.messages) {
      const consolidatedMessages = handlers.consolidateMessages(data.messages);
      handlers.setMessages(consolidatedMessages);
      if (data.app_url) {
        handlers.setAppUrl(data.app_url);
      }
      return;
    }

    // ── error ──
    if (data.e === "error") {
      const errorMessage = (data.message as string) || "An error occurred";
      handlers.setError(errorMessage);
      handlers.setIsBuilding(false);
      handlers.setIsSending(false);
      handlers.setBuildStage("completed");
      handlers.setBuildLogs?.([]);

      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [
            ...prev.slice(0, -1),
            { ...lastMsg, content: `⚠️ ${errorMessage}` },
          ];
        }
        return [
          ...prev,
          {
            id: Date.now().toString() + "-error",
            role: "assistant" as const,
            content: `⚠️ ${errorMessage}`,
            created_at: new Date().toISOString(),
            event_type: "error",
          },
        ];
      });
      return;
    }

    // ── chat_response (non-build replies) ──
    if (data.e === "chat_response") {
      const message = (data.message as string) || "";
      if (!message) return;

      handlers.setIsSending(false);
      handlers.setIsBuilding(false);
      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [...prev.slice(0, -1), { ...lastMsg, content: message }];
        }
        return [
          ...prev,
          {
            id: Date.now().toString() + "-chat",
            role: "assistant" as const,
            content: message,
            created_at: new Date().toISOString(),
            event_type: "chat_response",
          },
        ];
      });
      return;
    }
  } catch (err) {
    console.error("Failed to parse SSE message:", err);
  }
}
