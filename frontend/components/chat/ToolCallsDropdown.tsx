import { ChevronDown, Loader2 } from "lucide-react";

interface ToolCall {
  name: string;
  status: "success" | "error" | "running";
  output?: string;
}

interface ToolCallsDropdownProps {
  toolCalls: ToolCall[];
  isExpanded: boolean;
  onToggle: () => void;
}

export function ToolCallsDropdown({
  toolCalls,
  isExpanded,
  onToggle,
}: ToolCallsDropdownProps) {
  if (toolCalls.length === 0) return null;

  return (
    <div className="border-t border-border bg-secondary/30 px-4 py-2">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors py-2"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium">
            {toolCalls.length} tool{toolCalls.length !== 1 ? "s" : ""} used
          </span>
        </div>
        <ChevronDown
          size={16}
          className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
        />
      </button>

      {isExpanded && (
        <div className="mt-1 mb-2 space-y-1.5 max-h-56 overflow-y-auto">
          {toolCalls.map((tool, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 text-xs px-3 py-2.5 bg-card border border-border rounded-lg"
            >
              <div className="flex items-center justify-center w-5 h-5 rounded-full bg-secondary shrink-0 mt-0.5">
                {tool.status === "success" ? (
                  <span className="text-primary text-xs">✓</span>
                ) : tool.status === "error" ? (
                  <span className="text-destructive text-xs">✗</span>
                ) : (
                  <Loader2 size={12} className="animate-spin text-primary" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-foreground/90 font-medium mb-0.5">{tool.name}</div>
                {tool.output && (
                  <div className="text-muted-foreground text-[11px] leading-relaxed break-words">
                    {tool.output}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
