import type { SSEMessage, SSEHandlers } from "./chat-types";

function extractToolDetail(
  name: string,
  input: Record<string, any> | undefined,
): string {
  if (!input || typeof input !== "object") return "";

  switch (name) {
    case "create_file":
    case "read_file":
    case "write_file":
    case "delete_file":
      return input.file_path || input.path || "";
    case "execute_command":
      return input.command || "";
    case "list_directory":
      return input.path || input.directory || ".";
    case "write_multiple_files": {
      const raw = input.files;
      if (!raw) return "";
      try {
        const files = typeof raw === "string" ? JSON.parse(raw) : raw;
        if (Array.isArray(files)) {
          const paths = files
            .map((f: any) => f.path || f.file_path || "")
            .filter(Boolean);
          return paths.join(", ");
        }
      } catch { /* ignore */ }
      const str = typeof raw === "string" ? raw : JSON.stringify(raw);
      const matches = [...str.matchAll(/"path"\s*:\s*"([^"]+)"/g)];
      if (matches.length > 0) return matches.map((m) => m[1]).join(", ");
      return "";
    }
    default:
      return "";
  }
}

export function handleSSEMessage(event: MessageEvent, handlers: SSEHandlers) {
  try {
    const data: SSEMessage = JSON.parse(event.data);
    console.log("SSE:", data.e || data.type);

    if (data.e === "tool_started") {
      const toolName = (data.tool_name as string) || "Unknown Tool";
      const toolInput = data.tool_input as Record<string, any> | undefined;
      const detail = extractToolDetail(toolName, toolInput);
      const input = toolInput ? JSON.stringify(toolInput) : undefined;

      handlers.setCurrentTool({ name: toolName, status: "running" });

      handlers.setMessages((prev) => {
        if (prev.length === 0) return prev;
        const lastMsg = prev[prev.length - 1];
        if (lastMsg.role === "assistant") {
          return [
            ...prev.slice(0, -1),
            {
              ...lastMsg,
              tool_calls: [
                ...(lastMsg.tool_calls || []),
                { name: toolName, status: "running" as const, detail, input },
              ],
            },
          ];
        }
        return [
          ...prev,
          {
            id: Date.now().toString() + "-assistant",
            role: "assistant" as const,
            content: "",
            created_at: new Date().toISOString(),
            event_type: "tool_started",
            tool_calls: [{ name: toolName, status: "running" as const, detail, input }],
          },
        ];
      });
      return;
    }

    if (data.e === "tool_completed") {
      const toolName =
        (data.tool_name as string) || handlers.currentTool?.name || "Tool";
      const toolOutput = data.tool_output || "Completed";

      handlers.setCurrentTool(null);

      handlers.setMessages((prev) =>
        prev.map((msg) => {
          if (msg.role === "assistant" && msg.tool_calls) {
            const lastRunningIdx = msg.tool_calls
              .map((t, i) =>
                t.name === toolName && t.status === "running" ? i : -1,
              )
              .filter((i) => i >= 0)
              .pop();

            if (lastRunningIdx !== undefined && lastRunningIdx >= 0) {
              const updated = [...msg.tool_calls];
              updated[lastRunningIdx] = {
                ...updated[lastRunningIdx],
                status: "success" as const,
                output:
                  typeof toolOutput === "string"
                    ? toolOutput
                    : JSON.stringify(toolOutput),
              };
              return { ...msg, tool_calls: updated };
            }
          }
          return msg;
        }),
      );
      return;
    }

    if (
      data.e === "started" ||
      data.e === "builder_started" ||
      data.e === "workflow_started" ||
      data.e === "planner_started"
    ) {
      handlers.setIsBuilding(true);
    }

    if (data.url) {
      handlers.setIsBuilding(false);
      handlers.pollUrlUntilReady(data.url);
    }

    if (data.e === "completed" || data.e === "workflow_completed") {
      handlers.setIsBuilding(false);
      handlers.setCurrentTool(null);

      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [...prev.slice(0, -1), { ...lastMsg, isCompleted: true }];
        }
        return prev;
      });

      const updatedUser = localStorage.getItem("user_data");
      if (updatedUser) {
        handlers.setUserData(JSON.parse(updatedUser));
      }
    }

    if (data.e === "history" && data.messages) {
      const consolidatedMessages = handlers.consolidateMessages(data.messages);
      handlers.setMessages(consolidatedMessages);
      if (data.app_url) {
        handlers.setAppUrl(data.app_url);
      }
      return;
    }

    if (data.type === "error" || data.e === "error") {
      handlers.setError((data.message as string) || "An error occurred");
    }

    if (data.type === "token_update" || data.tokens_remaining !== undefined) {
      const user = JSON.parse(localStorage.getItem("user_data") || "{}");
      user.tokens_remaining = data.tokens_remaining;
      if (data.reset_in_hours !== undefined) {
        user.reset_in_hours = data.reset_in_hours;
      }
      localStorage.setItem("user_data", JSON.stringify(user));
      handlers.setUserData(user);
    }

    if (data.e === "thinking") {
      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") return prev;
        return [
          ...prev,
          {
            id: Date.now().toString() + "-assistant",
            role: "assistant" as const,
            content: "Analyzing your request...",
            created_at: new Date().toISOString(),
            event_type: "thinking",
          },
        ];
      });
      return;
    }

    if (
      data.e === "builder_complete" ||
      data.e === "code_validator_started" ||
      data.e === "code_validator_complete" ||
      data.e === "app_check_started" ||
      data.e === "app_check_complete"
    ) {
      return;
    }

    if (data.e === "description") {
      const description = (data.message as string) || "";
      if (!description) return;

      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [...prev.slice(0, -1), { ...lastMsg, content: description }];
        }
        return [
          ...prev,
          {
            id: Date.now().toString() + "-desc",
            role: "assistant" as const,
            content: description,
            created_at: new Date().toISOString(),
            event_type: "description",
          },
        ];
      });
      return;
    }

    if (data.e === "summary") {
      const summary = (data.message as string) || "";
      if (!summary) return;

      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [...prev.slice(0, -1), { ...lastMsg, summary }];
        }
        return prev;
      });
      return;
    }

    if (data.e === "planner_complete") {
      const plan = data.content || data.plan;
      let title = "Planning complete";

      if (plan && typeof plan === "object") {
        const overview =
          (plan as any).applicationOverview ||
          (plan as any).application_overview ||
          (plan as any).overview;
        if (overview) {
          const t =
            typeof overview === "object"
              ? (overview as any).title || (overview as any).purpose
              : String(overview);
          if (t) title = String(t).substring(0, 150);
        } else if ((plan as any).title) {
          title = String((plan as any).title).substring(0, 150);
        }
      }

      handlers.setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant") {
          return [...prev.slice(0, -1), { ...lastMsg, content: title }];
        }
        return [
          ...prev,
          {
            id: Date.now().toString() + "-plan",
            role: "assistant" as const,
            content: title,
            created_at: new Date().toISOString(),
            event_type: "planner_complete",
          },
        ];
      });
      return;
    }
  } catch (err) {
    console.error("Failed to parse SSE message:", err);
  }
}
