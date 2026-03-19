import { Button } from "@/components/ui/button";
import { ChevronLeft, Eye, EyeOff, Plus } from "lucide-react";
import type { UserData } from "@/api";
import { ApiKeySettings } from "./ApiKeySettings";

interface ChatIdHeaderProps {
  userData: UserData | null;
  showPreview: boolean;
  onTogglePreview: () => void;
  onNewChat: () => void;
  onBack: () => void;
  onUserDataUpdate?: (data: UserData) => void;
}

export function ChatIdHeader({
  userData,
  showPreview,
  onTogglePreview,
  onNewChat,
  onBack,
  onUserDataUpdate,
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
            <div className="hidden md:flex items-center px-3 py-1.5 rounded-lg bg-secondary border border-border text-sm">
              <span className="text-muted-foreground">{userData.email}</span>
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
          <ApiKeySettings userData={userData} onUserDataUpdate={onUserDataUpdate || (() => {})} />
          <Button size="sm" onClick={onNewChat} className="gap-1.5">
            <Plus size={14} />
            New Chat
          </Button>
        </div>
      </div>
    </div>
  );
}
