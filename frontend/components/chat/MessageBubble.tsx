import React, { useState, useEffect, useRef } from "react";
import { Loader2, CheckCircle2, AlertCircle, Clock } from "lucide-react";

interface ToolCall {
  name: string;
  status: "success" | "error" | "running";
  output?: string;
  detail?: string;
  input?: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  formatted?: string;
  event_type?: string;
  isCompleted?: boolean;
  isProgress?: boolean;
  summary?: string;
  buildDuration?: number;
  tool_calls?: ToolCall[];
}

interface MessageBubbleProps {
  message: Message;
  isLastMessage: boolean;
}

const TOOL_LABELS: Record<string, string> = {
  create_file: "Created",
  read_file: "Read",
  edit_file: "Edited",
  write_file: "Updated",
  delete_file: "Deleted",
  execute_command: "Ran",
  list_directory: "Listed",
  check_missing_packages: "Checked packages",
  get_context: "Loaded context",
  save_context: "Saved context",
  write_multiple_files: "Wrote",
};

const TOOL_RUNNING_LABELS: Record<string, string> = {
  create_file: "Creating",
  read_file: "Reading",
  edit_file: "Editing",
  write_file: "Updating",
  delete_file: "Deleting",
  execute_command: "Running",
  list_directory: "Listing",
  check_missing_packages: "Checking packages",
  get_context: "Loading context",
  save_context: "Saving context",
  write_multiple_files: "Writing",
};

function formatToolLabel(name: string, status: string): string {
  if (status === "running") {
    return TOOL_RUNNING_LABELS[name] || name.replace(/_/g, " ");
  }
  return TOOL_LABELS[name] || name.replace(/_/g, " ");
}

function parseDetailFromInput(name: string, input?: string): string {
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

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

/** Live elapsed-time counter — ticks every second while visible. */
function ElapsedTimer() {
  const startRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground ml-2 tabular-nums">
      <Clock size={11} className="shrink-0" />
      {formatDuration(elapsed)}
    </span>
  );
}

function getCleanContent(content: string): string {
  if (!content) return "";
  const trimmed = content.trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) return "";
  if (trimmed.includes("```")) return "";
  if (trimmed.includes('\\"') && trimmed.includes("\\n")) return "";
  if (trimmed.length > 1000) return "";
  return trimmed;
}

function renderInlineMarkdown(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <strong key={match.index} className="font-semibold text-foreground">
        {match[1]}
      </strong>,
    );
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? <>{parts}</> : text;
}

function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  lines.forEach((line, lineIdx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      elements.push(<div key={`br-${lineIdx}`} className="h-1.5" />);
      return;
    }
    if (trimmed.startsWith("- ")) {
      const content = trimmed.slice(2);
      elements.push(
        <div key={lineIdx} className="flex items-start gap-2 pl-1">
          <span className="text-muted-foreground mt-0.5">•</span>
          <span>{renderInlineMarkdown(content)}</span>
        </div>,
      );
      return;
    }
    elements.push(<div key={lineIdx}>{renderInlineMarkdown(trimmed)}</div>);
  });

  return elements;
}

export function MessageBubble({ message, isLastMessage }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-xl px-4 py-3 rounded-lg bg-primary text-primary-foreground">
          <p className="text-sm">{message.content}</p>
        </div>
      </div>
    );
  }

  const cleanContent = getCleanContent(message.content);
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;
  const hasSummary = !!message.summary;
  const showThinking = !cleanContent && !hasToolCalls && isLastMessage;
  const isError = message.event_type === "error";

  if (isError) {
    return (
      <div className="flex justify-start">
        <div className="max-w-2xl w-full bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertCircle size={15} className="text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-400">{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-2xl w-full bg-card border border-border rounded-lg p-4">
        {cleanContent && (
          <div className="text-sm text-foreground/80 mb-3 leading-relaxed">
            {message.isProgress && isLastMessage ? (
              <div className="flex items-center gap-2">
                <Loader2 size={13} className="animate-spin text-primary shrink-0" />
                <span>{cleanContent}</span>
                <ElapsedTimer />
              </div>
            ) : (
              renderMarkdown(cleanContent)
            )}
          </div>
        )}

        {hasToolCalls && (
          <div className="space-y-1">
            {message.tool_calls!.map((tool, idx) => {
              const detail =
                tool.detail || parseDetailFromInput(tool.name, tool.input);

              return (
                <div key={idx} className="flex items-center gap-2 py-0.5">
                  {tool.status === "running" ? (
                    <Loader2 size={13} className="animate-spin text-primary shrink-0" />
                  ) : tool.status === "success" ? (
                    <span className="text-primary text-sm shrink-0">✓</span>
                  ) : (
                    <span className="text-destructive text-sm shrink-0">✗</span>
                  )}
                  <span className="text-xs text-foreground/80">
                    {formatToolLabel(tool.name, tool.status)}
                  </span>
                  {detail && (
                    <span className="text-xs text-muted-foreground font-mono break-all">
                      {detail}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {hasSummary && (
          <div className="text-sm text-foreground/80 mt-3 pt-3 border-t border-border leading-relaxed">
            {renderMarkdown(message.summary!)}
          </div>
        )}

        {message.isCompleted && (
          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border">
            <CheckCircle2 size={15} className="text-primary shrink-0" />
            <span className="text-sm text-primary font-medium">
              Your app is ready
            </span>
            {message.buildDuration && (
              <span className="inline-flex items-center gap-1 text-xs text-muted-foreground ml-auto tabular-nums">
                <Clock size={11} className="shrink-0" />
                {formatDuration(message.buildDuration)}
              </span>
            )}
          </div>
        )}

        {showThinking && (
          <div className="flex items-center gap-2">
            <Loader2 size={13} className="animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Thinking...</span>
            <ElapsedTimer />
          </div>
        )}
      </div>
    </div>
  );
}
