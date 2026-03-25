import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ChevronLeft, Eye, EyeOff, Plus, Rocket, ExternalLink, Loader2 } from "lucide-react";
import type { UserData } from "@/api";
import { chatApi } from "@/api/chat";
import { ApiKeySettings } from "./ApiKeySettings";

interface ChatIdHeaderProps {
  userData: UserData | null;
  showPreview: boolean;
  onTogglePreview: () => void;
  onNewChat: () => void;
  onBack: () => void;
  onUserDataUpdate?: (data: UserData) => void;
  projectId: string;
  appUrl: string | null;
  deployedUrl: string | null;
  onDeployedUrl?: (url: string) => void;
}

export function ChatIdHeader({
  userData,
  showPreview,
  onTogglePreview,
  onNewChat,
  onBack,
  onUserDataUpdate,
  projectId,
  appUrl,
  deployedUrl,
  onDeployedUrl,
}: ChatIdHeaderProps) {
  const [isDeploying, setIsDeploying] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);

  const handleDeploy = async () => {
    setIsDeploying(true);
    setDeployError(null);
    try {
      const result = await chatApi.deployProject(projectId);
      onDeployedUrl?.(result.url);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Deploy failed";
      setDeployError(msg);
      setTimeout(() => setDeployError(null), 4000);
    } finally {
      setIsDeploying(false);
    }
  };

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

          {appUrl && (
            <Button
              size="sm"
              variant="outline"
              className="gap-1.5"
              onClick={handleDeploy}
              disabled={isDeploying}
            >
              {isDeploying ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Rocket size={14} />
              )}
              {isDeploying ? "Deploying..." : deployedUrl ? "Redeploy" : "Deploy"}
            </Button>
          )}
          {deployedUrl && (
            <Button
              size="sm"
              variant="outline"
              className="gap-1.5 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/10"
              onClick={() => window.open(deployedUrl, "_blank")}
            >
              <ExternalLink size={14} />
              Live Site
            </Button>
          )}

          {deployError && (
            <span className="text-xs text-red-400 max-w-48 truncate" title={deployError}>
              {deployError}
            </span>
          )}

          <Button size="sm" onClick={onNewChat} className="gap-1.5">
            <Plus size={14} />
            New Chat
          </Button>
        </div>
      </div>
    </div>
  );
}
