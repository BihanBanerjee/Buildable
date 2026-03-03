import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  return (
    <div className="border-t border-border bg-background/60 backdrop-blur-md p-4">
      <form onSubmit={onSubmit}>
        <div className="bg-card border border-border rounded-lg p-3 hover:border-emerald-900/50 transition-colors">
          <div className="flex gap-3">
            <Input
              type="text"
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              placeholder="Follow-up message..."
              className="flex-1 border-0 bg-transparent text-foreground placeholder:text-muted-foreground focus-visible:ring-0"
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
