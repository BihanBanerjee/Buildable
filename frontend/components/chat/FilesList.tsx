import { FileCode, Folder, ChevronRight, ChevronDown } from "lucide-react";
import { useState } from "react";

interface FilesListProps {
  files: string[];
  appUrl: string | null;
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
  depth = 0,
}: {
  node: FileNode;
  depth?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(depth === 0);

  return (
    <div>
      <div
        className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-secondary/60 cursor-pointer transition-colors"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => {
          if (node.isDirectory) setIsExpanded(!isExpanded);
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
            <FileTreeNode key={child.path} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function FilesList({ files }: FilesListProps) {
  const fileTree = buildFileTree(files);

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <FileCode className="w-10 h-10 mb-3 opacity-30" />
        <p className="text-sm">No files yet</p>
        <p className="text-xs mt-1 opacity-60">Files appear once your app is built</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="mb-4 pb-3 border-b border-border">
        <h3 className="text-foreground/90 font-semibold text-sm">Project Files</h3>
        <p className="text-muted-foreground text-xs mt-0.5">
          {files.length} file{files.length !== 1 ? "s" : ""}
        </p>
      </div>
      <div className="space-y-0.5">
        {fileTree.map((node) => (
          <FileTreeNode key={node.path} node={node} />
        ))}
      </div>
    </div>
  );
}
