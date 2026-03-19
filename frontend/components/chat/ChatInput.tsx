import { useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";

interface ChatInputProps {
  input: string;
  wsConnected: boolean;
  isBuilding: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
}

export function ChatInput({
  input,
  wsConnected,
  isBuilding,
  onInputChange,
  onSubmit,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }, []);

  useEffect(() => {
    autoResize();
  }, [input, autoResize]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && wsConnected && !isBuilding) {
        onSubmit(e as unknown as React.FormEvent);
      }
    }
  };

  return (
    <div className="border-t border-border bg-background/60 backdrop-blur-md p-4">
      <form onSubmit={onSubmit}>
        <div className="bg-card border border-border rounded-lg p-3 hover:border-emerald-900/50 transition-colors">
          <div className="flex gap-3 items-end">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Follow-up message..."
              rows={1}
              className="flex-1 resize-none border-0 bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none text-sm py-1"
              disabled={!wsConnected || isBuilding}
            />
            <Button
              type="submit"
              disabled={!wsConnected || !input.trim() || isBuilding}
              size="icon"
              className="rounded-lg w-8 h-8 shrink-0"
            >
              <ArrowUp size={16} />
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
}
