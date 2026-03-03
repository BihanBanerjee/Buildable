import { Button } from "@/components/ui/button";

export function PromotionBanner() {
  return (
    <div className="mt-24 px-4 py-3 rounded-xl border border-border bg-card max-w-2xl w-full flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="px-2 py-0.5 rounded-md bg-primary text-primary-foreground text-xs font-semibold">
          New
        </div>
        <span className="text-foreground/70 text-sm">
          Build faster with multi-model support — Gemini & Claude
        </span>
      </div>
      <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground shrink-0">
        Got it
      </Button>
    </div>
  );
}
