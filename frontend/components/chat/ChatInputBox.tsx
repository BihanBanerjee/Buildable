import { useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";

interface ChatInputBoxProps {
  input: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
}

export function ChatInputBox({
  input,
  isLoading,
  onInputChange,
  onSubmit,
}: ChatInputBoxProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, []);

  useEffect(() => {
    autoResize();
  }, [input, autoResize]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        onSubmit(e as unknown as React.FormEvent);
      }
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <div className="bg-card border border-border rounded-xl p-4 hover:border-emerald-900/50 transition-colors">
        <textarea
          ref={textareaRef}
          placeholder="Describe the app you want to build..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={1}
          className="w-full resize-none border-0 bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none text-base py-1"
        />
        <div className="flex items-center justify-end mt-3 pt-3 border-t border-border">
          <Button
            type="submit"
            disabled={isLoading || !input.trim()}
            size="icon"
            className="rounded-lg w-8 h-8"
          >
            <ArrowUp size={16} />
          </Button>
        </div>
      </div>
    </form>
  );
}
