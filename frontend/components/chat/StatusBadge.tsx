export function StatusBadge() {
  return (
    <div className="mb-8 flex items-center gap-2 px-4 py-2 rounded-full border border-emerald-500/20 bg-emerald-500/8">
      <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
      <span className="text-sm text-foreground/80">AI Online — Ready to build</span>
    </div>
  );
}
