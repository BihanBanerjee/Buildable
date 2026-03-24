import { Eye, FileCode, Globe, ExternalLink, RefreshCw, Loader2 } from "lucide-react";
import { useState, useCallback } from "react";
import { FileViewer } from "./FileViewer";

interface PreviewPanelProps {
  appUrl: string | null;
  isCheckingUrl: boolean;
  previewWidth: number;
  files: string[];
  projectId: string;
  sandboxActive: boolean;
  isRestartingSandbox: boolean;
  onRestartSandbox: () => void;
  isLoadingFiles?: boolean;
}

type TabType = "preview" | "files";

export function PreviewPanel({
  appUrl,
  isCheckingUrl,
  previewWidth,
  files,
  projectId,
  sandboxActive,
  isRestartingSandbox,
  onRestartSandbox,
  isLoadingFiles = false,
}: PreviewPanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>("preview");
  const [iframeLoaded, setIframeLoaded] = useState(false);

  const handleIframeLoad = useCallback(() => setIframeLoaded(true), []);

  return (
    <div
      className="flex flex-col bg-background border-l border-border"
      style={{ width: `${previewWidth}%` }}
    >
      {/* Tab bar */}
      <div className="flex items-center justify-between border-b border-border bg-secondary/20">
        <div className="flex">
          <button
            onClick={() => setActiveTab("preview")}
            className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors ${
              activeTab === "preview"
                ? "text-foreground border-b-2 border-primary bg-card"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary/40"
            }`}
          >
            <Globe className="w-3.5 h-3.5" />
            Preview
          </button>
          <button
            onClick={() => setActiveTab("files")}
            className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors ${
              activeTab === "files"
                ? "text-foreground border-b-2 border-primary bg-card"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary/40"
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Files {files.length > 0 && `(${files.length})`}
          </button>
        </div>

        {appUrl && !isCheckingUrl && (
          <button
            onClick={() => window.open(appUrl, "_blank")}
            className="flex items-center gap-1.5 px-3 py-2 mx-2 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
            title="Open in new tab"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Open
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "preview" ? (
          <div className="h-full p-4">
            {isCheckingUrl ? (
              <div className="w-full h-full rounded-xl border border-border flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-10 w-10 border-2 border-primary border-t-transparent mx-auto mb-4" />
                  <p className="text-muted-foreground text-sm">
                    Waiting for app to start...
                  </p>
                </div>
              </div>
            ) : isRestartingSandbox ? (
              <div className="w-full h-full rounded-xl border border-border flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-10 w-10 border-2 border-primary border-t-transparent mx-auto mb-4" />
                  <p className="text-muted-foreground text-sm font-medium">
                    Restarting sandbox...
                  </p>
                  <p className="text-muted-foreground/60 text-xs mt-1">
                    Restoring files and starting Vite
                  </p>
                </div>
              </div>
            ) : appUrl && !sandboxActive ? (
              <div className="w-full h-full rounded-xl border border-border flex items-center justify-center">
                <div className="text-center">
                  <RefreshCw className="w-10 h-10 text-muted-foreground/20 mx-auto mb-3" />
                  <p className="text-foreground/80 text-sm font-medium mb-1">
                    Preview has expired
                  </p>
                  <p className="text-muted-foreground/60 text-xs mb-4">
                    The sandbox timed out after 30 minutes of inactivity
                  </p>
                  <button
                    onClick={onRestartSandbox}
                    className="flex items-center gap-2 px-4 py-2 text-xs font-medium bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors mx-auto"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Restart Preview
                  </button>
                </div>
              </div>
            ) : appUrl ? (
              <div className="w-full h-full rounded-xl overflow-hidden border border-border relative">
                {!iframeLoaded && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
                    <div className="text-center">
                      <Loader2 className="w-6 h-6 animate-spin text-primary mx-auto mb-3" />
                      <p className="text-muted-foreground text-sm">Loading preview...</p>
                    </div>
                  </div>
                )}
                <iframe
                  src={appUrl}
                  title="App Preview"
                  className="w-full h-full"
                  sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
                  onLoad={handleIframeLoad}
                />
              </div>
            ) : (
              <div className="w-full h-full rounded-xl border border-border flex items-center justify-center">
                <div className="text-center">
                  <Eye className="w-10 h-10 text-muted-foreground/20 mx-auto mb-3" />
                  <p className="text-muted-foreground text-sm">
                    Preview will appear here
                  </p>
                  <p className="text-muted-foreground/60 text-xs mt-1">
                    Build something to see it live
                  </p>
                </div>
              </div>
            )}
          </div>
        ) : isLoadingFiles ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mb-3" />
            <p className="text-sm">Loading files...</p>
          </div>
        ) : (
          <FileViewer files={files} projectId={projectId} />
        )}
      </div>
    </div>
  );
}
