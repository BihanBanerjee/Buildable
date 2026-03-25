"use client";

import { useEffect, useRef, useState } from "react";
import { Clock } from "lucide-react";

interface TerminalLogProps {
  logs: string[];
  isBuilding: boolean;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

function classifyLine(line: string): "file" | "error" | "success" | "info" {
  const trimmed = line.trim().toLowerCase();
  if (trimmed.startsWith("→") || trimmed.startsWith("->")) return "file";
  if (/error|failed|timed out/i.test(trimmed)) return "error";
  if (/generated|created|complete|ready|installed|started/i.test(trimmed)) return "success";
  return "info";
}

export function TerminalLog({ logs, isBuilding }: TerminalLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const startRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);
  const [dots, setDots] = useState("");

  // Reset timer when a new build starts
  useEffect(() => {
    if (isBuilding && logs.length <= 1) {
      startRef.current = Date.now();
      setElapsed(0);
    }
  }, [isBuilding, logs.length]);

  // Elapsed timer
  useEffect(() => {
    if (!isBuilding) return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isBuilding]);

  // Animated dots
  useEffect(() => {
    if (!isBuilding) return;
    const id = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 500);
    return () => clearInterval(id);
  }, [isBuilding]);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [logs]);

  if (!isBuilding && logs.length === 0) return null;

  const visibleLogs = logs.slice(-16);

  // Progress bar: grows with log count, max 95%
  const progressWidth = Math.min(95, 15 + logs.length * 6);

  return (
    <div className="border-t border-border bg-background/60 backdrop-blur-sm">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-emerald-500/70" />
            <div className="h-2 w-2 rounded-full bg-emerald-600/50" />
            <div className="h-2 w-2 rounded-full bg-emerald-700/40" />
          </div>
          <span className="text-[11px] font-mono text-muted-foreground">
            buildable@agent
          </span>
        </div>
        {isBuilding && (
          <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground tabular-nums">
            <Clock size={10} className="shrink-0" />
            {formatElapsed(elapsed)}
          </span>
        )}
      </div>

      {/* Progress bar */}
      {isBuilding && (
        <div className="h-0.5 bg-border/30 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-500/60 to-emerald-400/40 transition-all duration-700 ease-out"
            style={{ width: `${progressWidth}%` }}
          />
        </div>
      )}

      {/* Terminal body */}
      <div
        ref={scrollRef}
        className="px-4 py-2.5 overflow-y-auto font-mono text-xs leading-relaxed"
        style={{ maxHeight: "180px" }}
      >
        {visibleLogs.map((line, i) => {
          const type = classifyLine(line);
          const trimmed = line.trim();

          let colorClass = "text-muted-foreground";
          let prefix = "";

          if (type === "error") {
            colorClass = "text-red-400";
            prefix = "\u2716 ";
          } else if (type === "success") {
            colorClass = "text-emerald-400";
            prefix = "\u2714 ";
          } else if (type === "file") {
            colorClass = "text-foreground/70";
          } else {
            colorClass = "text-muted-foreground";
          }

          return (
            <div key={`${i}-${trimmed.slice(0, 20)}`} className={`${colorClass} py-px`}>
              {prefix}{trimmed}
            </div>
          );
        })}

        {isBuilding && (
          <div className="text-muted-foreground/60 py-px">
            Working{dots}
          </div>
        )}
      </div>
    </div>
  );
}
