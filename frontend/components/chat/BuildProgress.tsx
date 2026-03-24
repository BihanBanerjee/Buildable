import { useEffect, useRef, useState } from "react";
import { Loader2, CheckCircle2, Circle, Clock } from "lucide-react";
import type { BuildStage } from "@/lib/chat-types";

interface BuildProgressProps {
  currentStage: BuildStage | null;
}

const STAGES: { key: BuildStage; label: string }[] = [
  { key: "enhancer", label: "Understanding" },
  { key: "planner", label: "Planning" },
  { key: "builder", label: "Building" },
  { key: "validator", label: "Validating" },
  { key: "app_check", label: "Testing" },
];

function getStageIndex(stage: BuildStage | null): number {
  if (!stage) return -1;
  if (stage === "completed") return STAGES.length;
  return STAGES.findIndex((s) => s.key === stage);
}

function StageTimer() {
  const startRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)),
      1000,
    );
    return () => clearInterval(id);
  }, []);

  if (elapsed < 2) return null;

  return (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground tabular-nums ml-1">
      <Clock size={10} className="shrink-0" />
      {elapsed}s
    </span>
  );
}

export function BuildProgress({ currentStage }: BuildProgressProps) {
  if (!currentStage) return null;

  const activeIdx = getStageIndex(currentStage);

  return (
    <div className="flex items-center gap-1 px-4 py-2.5 border-t border-border bg-background/40 backdrop-blur-sm">
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < activeIdx;
        const isActive = idx === activeIdx;

        return (
          <div key={stage.key} className="flex items-center gap-1">
            {/* Step indicator */}
            <div className="flex items-center gap-1.5">
              {isCompleted ? (
                <CheckCircle2 size={14} className="text-primary shrink-0" />
              ) : isActive ? (
                <Loader2
                  size={14}
                  className="animate-spin text-primary shrink-0"
                />
              ) : (
                <Circle
                  size={14}
                  className="text-muted-foreground/40 shrink-0"
                />
              )}
              <span
                className={`text-xs font-medium transition-colors ${
                  isCompleted
                    ? "text-primary"
                    : isActive
                      ? "text-foreground"
                      : "text-muted-foreground/40"
                }`}
              >
                {stage.label}
              </span>
              {isActive && <StageTimer />}
            </div>

            {/* Connector line */}
            {idx < STAGES.length - 1 && (
              <div
                className={`w-4 h-px mx-0.5 transition-colors ${
                  isCompleted ? "bg-primary" : "bg-muted-foreground/20"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
