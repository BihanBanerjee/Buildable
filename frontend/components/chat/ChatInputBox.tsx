import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  return (
    <form onSubmit={onSubmit}>
      <div className="bg-card border border-border rounded-xl p-4 hover:border-emerald-900/50 transition-colors">
        <Input
          type="text"
          placeholder="Describe the app you want to build..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          disabled={isLoading}
          className="border-0 bg-transparent text-foreground placeholder:text-muted-foreground focus-visible:ring-0 text-base h-auto py-1"
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
