import { Button } from "@/components/ui/button";
import { ChevronLeft, Eye, EyeOff, Plus } from "lucide-react";
import type { UserData } from "@/api";

interface ChatIdHeaderProps {
  userData: UserData | null;
  showPreview: boolean;
  onTogglePreview: () => void;
  onNewChat: () => void;
  onBack: () => void;
}

export function ChatIdHeader({
  userData,
  showPreview,
  onTogglePreview,
  onNewChat,
  onBack,
}: ChatIdHeaderProps) {
  return (
    <div className="border-b border-border px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onBack}
            className="text-muted-foreground hover:text-foreground"
          >
            <ChevronLeft size={20} />
          </Button>
          <span className="font-semibold text-foreground text-sm tracking-tight">
            Buildable
          </span>
        </div>

        <div className="flex items-center gap-2">
          {userData && (
            <div
              className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary border border-border text-sm"
              title={
                !userData.is_unlimited && userData.reset_in_hours !== undefined
                  ? `Resets in ${userData.reset_in_hours.toFixed(1)}h`
                  : undefined
              }
            >
              <span className="text-muted-foreground">{userData.email}</span>
              <span className="text-border">•</span>
              <span
                className={`font-medium ${
                  userData.is_unlimited
                    ? "text-foreground"
                    : userData.tokens_remaining === 0
                      ? "text-red-400"
                      : userData.tokens_remaining <= 2
                        ? "text-yellow-400"
                        : "text-foreground"
                }`}
              >
                {userData.is_unlimited ? "∞ unlimited" : `${userData.tokens_remaining} tokens`}
              </span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onTogglePreview}
            className="text-muted-foreground hover:text-foreground hidden md:flex"
            title={showPreview ? "Hide preview" : "Show preview"}
          >
            {showPreview ? <EyeOff size={18} /> : <Eye size={18} />}
          </Button>
          <Button size="sm" onClick={onNewChat} className="gap-1.5">
            <Plus size={14} />
            New Chat
          </Button>
        </div>
      </div>
    </div>
  );
}
