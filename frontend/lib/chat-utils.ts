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
    "thinking",
    "tool_started",
    "tool_completed",
    "planner_complete",
    "builder_complete",
    "validator_complete",
    "description",
    "summary",
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
        if (currentAssistantGroup) {
          currentAssistantGroup.isCompleted = true;
        }
        continue;
      }

      if (msg.event_type && filterOutEvents.includes(msg.event_type)) {
        continue;
      }

      if (!msg.event_type || assistantEventTypes.includes(msg.event_type)) {
        const isRichContent =
          msg.event_type === "description" || msg.event_type === "summary";
        const keepContent = isRichContent
          ? msg.content.trim()
          : isCleanContent(msg.content)
            ? msg.content.trim()
            : "";

        if (currentAssistantGroup) {
          if (msg.event_type === "description" && keepContent) {
            currentAssistantGroup.content = keepContent;
          } else if (msg.event_type === "summary" && keepContent) {
            currentAssistantGroup.summary = keepContent;
          } else if (keepContent) {
            currentAssistantGroup.content = currentAssistantGroup.content
              ? currentAssistantGroup.content + "\n" + keepContent
              : keepContent;
          }
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            if (!currentAssistantGroup.tool_calls) {
              currentAssistantGroup.tool_calls = [];
            }
            currentAssistantGroup.tool_calls.push(...msg.tool_calls);
          }
        } else {
          currentAssistantGroup = {
            ...msg,
            content: keepContent,
            event_type: "thinking",
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
