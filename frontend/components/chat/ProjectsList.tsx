"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { chatApi } from "@/api/chat";
import type { Project } from "@/api/types";
import { FolderOpen, Clock, Loader2, Trash2 } from "lucide-react";

interface ProjectsListProps {
  trigger?: React.ReactNode;
}

export function ProjectsList({ trigger }: ProjectsListProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (isOpen) {
      loadProjects();
    }
  }, [isOpen]);

  const loadProjects = async () => {
    setIsLoading(true);
    try {
      const response = await chatApi.listProjects();
      setProjects(response.projects);
    } catch (error) {
      console.error("Failed to load projects:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProjectClick = (projectId: string) => {
    setIsOpen(false);
    router.push(`/chat/${projectId}`);
  };

  const handleDeleteClick = (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation();
    setConfirmingId(projectId);
  };

  const handleDeleteCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmingId(null);
  };

  const handleDeleteConfirm = async (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation();
    setConfirmingId(null);
    setDeletingId(projectId);
    try {
      await chatApi.deleteProject(projectId);
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      if (window.location.pathname.includes(projectId)) {
        router.push("/chat");
      }
    } catch (error) {
      console.error("Failed to delete project:", error);
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        {trigger || (
          <button className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg transition-colors">
            <FolderOpen size={15} />
            <span>Projects</span>
          </button>
        )}
      </SheetTrigger>
      <SheetContent className="h-full w-[380px] sm:w-[480px]">
        <SheetHeader>
          <SheetTitle>My Projects</SheetTitle>
          <SheetDescription>View and open your previous builds</SheetDescription>
        </SheetHeader>

        <div className="p-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-5 h-5 text-muted-foreground animate-spin" />
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-12">
              <FolderOpen className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
              <p className="text-muted-foreground text-sm">No projects yet</p>
              <p className="text-muted-foreground/60 text-xs mt-1">
                Start a new chat to create your first project
              </p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[calc(100vh-100px)] overflow-y-auto pr-1">
              {projects.map((project) => (
                <div
                  key={project.id}
                  className="group relative w-full text-left p-4 rounded-lg bg-secondary/50 hover:bg-secondary border border-border hover:border-emerald-900/40 transition-all cursor-pointer"
                  onClick={() => confirmingId === project.id ? null : handleProjectClick(project.id)}
                >
                  {confirmingId === project.id ? (
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm text-foreground/80">Delete this project?</p>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={handleDeleteCancel}
                          className="px-2.5 py-1 text-xs rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={(e) => handleDeleteConfirm(e, project.id)}
                          className="px-2.5 py-1 text-xs rounded-md bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-foreground font-medium text-sm truncate mb-1">
                          {project.title}
                        </h3>
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <Clock size={11} />
                          <span>{formatDate(project.created_at)}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => handleDeleteClick(e, project.id)}
                          disabled={deletingId === project.id}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded text-muted-foreground hover:text-red-400 hover:bg-red-500/10 transition-all"
                          title="Delete project"
                        >
                          {deletingId === project.id ? (
                            <Loader2 size={14} className="animate-spin" />
                          ) : (
                            <Trash2 size={14} />
                          )}
                        </button>
                        <span className="text-muted-foreground text-sm">→</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
