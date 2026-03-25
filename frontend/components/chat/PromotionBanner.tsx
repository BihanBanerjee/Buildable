import { Sparkles } from "lucide-react";

const EXAMPLES = [
  { label: "Todo App", prompt: "Build a beautiful todo app with categories, due dates, and a progress tracker" },
  { label: "Portfolio Site", prompt: "Create a minimal developer portfolio with a hero section, projects grid, and contact form" },
  { label: "Dashboard", prompt: "Build an analytics dashboard with stat cards, a line chart, and a recent activity table" },
  { label: "Landing Page", prompt: "Design a SaaS landing page with a hero, feature grid, pricing cards, and a FAQ accordion" },
];

interface ExamplePromptsProps {
  onSelect: (prompt: string) => void;
}

export function ExamplePrompts({ onSelect }: ExamplePromptsProps) {
  return (
    <div className="mt-10 max-w-2xl w-full">
      <div className="flex items-center gap-1.5 justify-center mb-3">
        <Sparkles size={13} className="text-muted-foreground/60" />
        <span className="text-xs text-muted-foreground/60">Try an example</span>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => onSelect(ex.prompt)}
            className="px-3 py-1.5 rounded-lg border border-border bg-card/50 text-sm text-muted-foreground hover:text-foreground hover:border-primary/30 hover:bg-card transition-colors"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
