import type { Message } from "./chat-types";

type ToolCall = NonNullable<Message["tool_calls"]>[number];

function isCleanContent(content: string): boolean {
  if (!content || !content.trim()) return false;
  const c = content.trim();
  if (c.startsWith("{") || c.startsWith("[")) return false;
  if (c.includes("```")) return false;
  if (c.length > 300) return false;
  if (c.includes('\\"') && c.includes("\\n")) return false;
  return true;
}

function extractDetailFromInput(name: string, input?: string): string {
  if (!input) return "";
  try {
    const parsed = JSON.parse(input);
    if (parsed.file_path) return parsed.file_path;
    if (parsed.path) return parsed.path;
    if (parsed.command) return parsed.command;
    if (name === "write_multiple_files" && parsed.files) {
      try {
        const files =
          typeof parsed.files === "string"
            ? JSON.parse(parsed.files)
            : parsed.files;
        if (Array.isArray(files)) {
          return files
            .map((f: any) => f.path || f.file_path || "")
            .filter(Boolean)
            .join(", ");
        }
      } catch { /* ignore */ }
      const str = typeof parsed.files === "string" ? parsed.files : JSON.stringify(parsed.files);
      const matches = [...str.matchAll(/"path"\s*:\s*"([^"]+)"/g)];
      if (matches.length > 0) return matches.map((m) => m[1]).join(", ");
    }
    return "";
  } catch {
    if (name === "write_multiple_files") {
      const matches = [...input.matchAll(/['"]path['"]\s*:\s*['"]([^'"]+)['"]/g)];
      if (matches.length > 0) return matches.map((m) => m[1]).join(", ");
    }
    const pathMatch = input.match(/['"](?:file_path|path)['"]\s*:\s*['"]([^'"]+)['"]/);
    if (pathMatch) return pathMatch[1];
    const cmdMatch = input.match(/['"]command['"]\s*:\s*['"]([^'"]+)['"]/);
    if (cmdMatch) return cmdMatch[1];
    return "";
  }
}

function deduplicateToolCalls(toolCalls: ToolCall[]): ToolCall[] {
  const result: ToolCall[] = [];

  for (let i = 0; i < toolCalls.length; i++) {
    const current = toolCalls[i];
    const next = toolCalls[i + 1];

    if (
      current.status === "running" &&
      next &&
      next.name === current.name &&
      next.status === "success"
    ) {
      const detail =
        next.detail ||
        current.detail ||
        extractDetailFromInput(current.name, current.input || next.input);
      toolCalls[i + 1] = { ...next, detail };
      continue;
    }

    if (!current.detail && current.input) {
      current.detail = extractDetailFromInput(current.name, current.input);
    }

    result.push(current);
  }

  return result;
}

export function consolidateMessages(msgs: Message[]): Message[] {
  const consolidated: Message[] = [];
  let currentAssistantGroup: Message | null = null;

  const assistantEventTypes = [
    "log",
    "token",
    "status",
    "warning",
    "chat_response",
  ];

  const filterOutEvents = ["started"];

  for (const msg of msgs) {
    if (msg.role === "user") {
      if (currentAssistantGroup) {
        consolidated.push(currentAssistantGroup);
        currentAssistantGroup = null;
      }
      consolidated.push(msg);
      continue;
    }

    if (msg.role === "assistant") {
      if (msg.event_type === "completed") {
        // If no preceding assistant message exists, create one from the completed message
        if (!currentAssistantGroup) {
          currentAssistantGroup = {
            ...msg,
            content: "",
          };
        }
        currentAssistantGroup.isCompleted = true;
        currentAssistantGroup.isSuccess = true;
        // Extract build summary from stored completed message
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          const summary = msg.tool_calls.find((t) => t.name === "build_summary");
          if (summary?.output) {
            try {
              const meta = JSON.parse(summary.output);
              if (meta.duration_s) currentAssistantGroup.buildDuration = meta.duration_s;
              if (meta.files) currentAssistantGroup.buildFiles = meta.files;
            } catch { /* ignore parse errors */ }
          }
        }
        continue;
      }

      if (msg.event_type && filterOutEvents.includes(msg.event_type)) {
        continue;
      }

      if (!msg.event_type || assistantEventTypes.includes(msg.event_type)) {
        const keepContent = msg.content?.trim() || "";

        if (currentAssistantGroup) {
          if (keepContent) {
            currentAssistantGroup.content = currentAssistantGroup.content
              ? currentAssistantGroup.content + "\n" + keepContent
              : keepContent;
          }
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            if (!currentAssistantGroup.tool_calls) {
              currentAssistantGroup.tool_calls = [];
            }
            // Default missing status to "success" (DB-stored tool_calls may lack status)
            const normalized = msg.tool_calls.map((t) => ({
              ...t,
              status: t.status || ("success" as const),
            }));
            currentAssistantGroup.tool_calls.push(...normalized);
          }
        } else {
          currentAssistantGroup = {
            ...msg,
            content: keepContent,
          };
        }
      } else {
        if (currentAssistantGroup) {
          consolidated.push(currentAssistantGroup);
          currentAssistantGroup = null;
        }
        consolidated.push(msg);
      }
    }
  }

  if (currentAssistantGroup) {
    consolidated.push(currentAssistantGroup);
  }

  return consolidated.map((msg) => {
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      return { ...msg, tool_calls: deduplicateToolCalls(msg.tool_calls) };
    }
    return msg;
  });
}

export function getAllToolCalls(messages: Message[]) {
  const allTools: ToolCall[] = [];
  messages.forEach((msg) => {
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      allTools.push(...msg.tool_calls);
    }
  });
  return allTools;
}
