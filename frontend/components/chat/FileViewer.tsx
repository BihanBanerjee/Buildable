"use client";

import { useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import {
  FileCode,
  Folder,
  ChevronRight,
  ChevronDown,
  Download,
  Loader2,
  FolderArchive,
} from "lucide-react";
import apiClient from "@/api/client";

interface FileViewerProps {
  files: string[];
  projectId: string;
}

interface FileNode {
  name: string;
  path: string;
  isDirectory: boolean;
  children?: FileNode[];
}

function buildFileTree(files: string[]): FileNode[] {
  const root: FileNode[] = [];

  files.forEach((filePath) => {
    const parts = filePath.split("/");
    let currentLevel = root;
    let currentPath = "";

    parts.forEach((part, index) => {
      currentPath += (index === 0 ? "" : "/") + part;
      const isLastPart = index === parts.length - 1;

      let existingNode = currentLevel.find((node) => node.name === part);

      if (!existingNode) {
        existingNode = {
          name: part,
          path: currentPath,
          isDirectory: !isLastPart,
          children: !isLastPart ? [] : undefined,
        };
        currentLevel.push(existingNode);
      }

      if (!isLastPart && existingNode.children) {
        currentLevel = existingNode.children;
      }
    });
  });

  return root;
}

function getLanguageFromPath(filePath: string): string {
  const ext = filePath.split(".").pop()?.toLowerCase();
  const languageMap: Record<string, string> = {
    js: "javascript",
    jsx: "javascript",
    ts: "typescript",
    tsx: "typescript",
    json: "json",
    html: "html",
    css: "css",
    scss: "scss",
    py: "python",
    md: "markdown",
    yml: "yaml",
    yaml: "yaml",
    xml: "xml",
    sh: "shell",
  };
  return languageMap[ext || ""] || "plaintext";
}

function getFileIcon(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase();
  const colorMap: Record<string, string> = {
    tsx: "text-blue-400",
    ts: "text-blue-400",
    jsx: "text-cyan-400",
    js: "text-yellow-400",
    css: "text-pink-400",
    json: "text-primary",
    html: "text-orange-400",
    md: "text-muted-foreground",
  };
  return <FileCode className={`w-4 h-4 ${colorMap[ext || ""] || "text-muted-foreground"}`} />;
}

function FileTreeNode({
  node,
  onSelectFile,
  selectedFile,
  depth = 0,
}: {
  node: FileNode;
  onSelectFile: (path: string) => void;
  selectedFile: string | null;
  depth?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(depth === 0);
  const isSelected = selectedFile === node.path;

  return (
    <div>
      <div
        className={`flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors ${
          isSelected
            ? "bg-primary/15 border-l-2 border-primary"
            : "hover:bg-secondary/60"
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => {
          if (node.isDirectory) {
            setIsExpanded(!isExpanded);
          } else {
            onSelectFile(node.path);
          }
        }}
      >
        {node.isDirectory ? (
          <>
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
            )}
            <Folder className="w-3.5 h-3.5 text-yellow-500/80" />
            <span className="text-sm text-foreground/80 font-medium">{node.name}</span>
          </>
        ) : (
          <>
            <div className="w-3.5" />
            {getFileIcon(node.name)}
            <span className="text-sm text-foreground/70">{node.name}</span>
          </>
        )}
      </div>

      {node.isDirectory && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <FileTreeNode
              key={child.path}
              node={child}
              onSelectFile={onSelectFile}
              selectedFile={selectedFile}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function FileViewer({ files, projectId }: FileViewerProps) {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [isLoadingFile, setIsLoadingFile] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const fileTree = buildFileTree(files);

  const fetchFileContent = async (filePath: string) => {
    setIsLoadingFile(true);
    try {
      const response = await apiClient.get<{ content: string }>(
        `/projects/${projectId}/files/${encodeURIComponent(filePath)}`,
      );
      setFileContent(response.data.content);
    } catch (error) {
      console.error("Failed to fetch file content:", error);
      setFileContent("// Error loading file content");
    } finally {
      setIsLoadingFile(false);
    }
  };

  const handleSelectFile = (filePath: string) => {
    setSelectedFile(filePath);
    fetchFileContent(filePath);
  };

  const handleDownloadFile = async () => {
    if (!selectedFile) return;
    try {
      const blob = new Blob([fileContent], { type: "text/plain" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = selectedFile.split("/").pop() || "file.txt";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download file:", error);
    }
  };

  const handleDownloadAll = async () => {
    setIsDownloading(true);
    try {
      const response = await apiClient.get<Blob>(`/projects/${projectId}/download`, {
        responseType: "blob",
      });
      const blob = response.data as Blob;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${projectId}-files.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download all files:", error);
    } finally {
      setIsDownloading(false);
    }
  };

  useEffect(() => {
    if (files.length > 0 && !selectedFile) {
      const firstFile =
        files.find((f) => !f.includes("/") || f.split("/").length === 1) ||
        files[0];
      handleSelectFile(firstFile);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <FileCode className="w-10 h-10 mb-3 opacity-30" />
        <p className="text-sm">No files yet</p>
        <p className="text-xs mt-1 text-muted-foreground/60">Files appear once your app is built</p>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* File Tree */}
      <div className="w-56 border-r border-border overflow-y-auto bg-secondary/20">
        <div className="sticky top-0 bg-card border-b border-border p-3 z-10">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-foreground/90 font-semibold text-xs uppercase tracking-wide">
              Files
            </h3>
            <button
              onClick={handleDownloadAll}
              disabled={isDownloading}
              className="flex items-center gap-1 px-2 py-0.5 text-xs bg-primary/15 hover:bg-primary/25 text-primary rounded transition-colors disabled:opacity-50"
              title="Download all as ZIP"
            >
              {isDownloading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <FolderArchive className="w-3 h-3" />
              )}
              ZIP
            </button>
          </div>
          <p className="text-muted-foreground text-xs">
            {files.length} file{files.length !== 1 ? "s" : ""}
          </p>
        </div>
        <div className="p-2">
          {fileTree.map((node) => (
            <FileTreeNode
              key={node.path}
              node={node}
              onSelectFile={handleSelectFile}
              selectedFile={selectedFile}
            />
          ))}
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 flex flex-col">
        {selectedFile ? (
          <>
            <div className="flex items-center justify-between px-4 py-2 bg-secondary/20 border-b border-border">
              <div className="flex items-center gap-2">
                {getFileIcon(selectedFile)}
                <span className="text-sm text-foreground/80 font-mono">{selectedFile}</span>
              </div>
              <button
                onClick={handleDownloadFile}
                className="flex items-center gap-1.5 px-3 py-1 text-xs bg-secondary hover:bg-secondary/70 text-muted-foreground hover:text-foreground rounded transition-colors"
              >
                <Download className="w-3 h-3" />
                Download
              </button>
            </div>
            <div className="flex-1 relative">
              {isLoadingFile ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <Editor
                  height="100%"
                  language={getLanguageFromPath(selectedFile)}
                  value={fileContent}
                  theme="vs-dark"
                  options={{
                    readOnly: true,
                    minimap: { enabled: false },
                    fontSize: 13,
                    lineNumbers: "on",
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    wordWrap: "on",
                    padding: { top: 12, bottom: 12 },
                  }}
                  loading={
                    <div className="flex items-center justify-center h-full">
                      <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                    </div>
                  }
                />
              )}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <FileCode className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Select a file to view</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
