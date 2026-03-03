import Link from "next/link";
import { ArrowRight, Zap, Eye, Download } from "lucide-react";

export default function Home() {
  return (
    <div
      className="min-h-screen w-full relative overflow-hidden"
      style={{ backgroundColor: "#030712" }}
    >
      {/* Emerald spotlight at top */}
      <div
        className="absolute inset-x-0 top-0 h-[600px] z-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 55% at 50% 0%, rgba(16,185,129,0.14) 0%, rgba(5,150,105,0.06) 50%, transparent 75%)",
        }}
      />

      {/* Nav */}
      <nav className="relative z-20 border-b border-emerald-900/25">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="font-semibold text-foreground tracking-tight">Buildable</span>
          <div className="flex items-center gap-3">
            <Link
              href="/signin"
              className="text-sm text-muted-foreground hover:text-foreground transition"
            >
              Sign In
            </Link>
            <Link
              href="/signup"
              className="px-4 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-emerald-600 transition"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative z-10">
        {/* Hero */}
        <section className="pt-24 pb-20 px-6">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/25 bg-emerald-500/8 text-emerald-400 text-xs font-medium mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              AI Web App Builder
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-foreground mb-6 leading-[1.1] tracking-tight">
              Describe it.{" "}
              <span className="text-primary">Build it.</span>{" "}
              Ship it.
            </h1>

            <p className="text-lg text-muted-foreground mb-10 max-w-xl mx-auto leading-relaxed">
              Turn plain English into production-ready React applications.
              Our AI agents plan, build, and validate your app — in minutes.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <Link
                href="/chat"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-emerald-600 transition group"
              >
                Start building free
                <ArrowRight size={16} className="group-hover:translate-x-0.5 transition" />
              </Link>
              <Link
                href="/signin"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-border text-foreground text-sm font-medium hover:bg-secondary transition"
              >
                Sign in
              </Link>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="py-20 px-6 border-t border-border/50">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-14">
              <h2 className="text-3xl font-bold text-foreground mb-3">How it works</h2>
              <p className="text-muted-foreground">Three steps from idea to deployed app</p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  step: "01",
                  icon: <span className="text-xl">💬</span>,
                  title: "Describe your app",
                  body: "Type what you want to build in plain English. No technical specs required.",
                },
                {
                  step: "02",
                  icon: <Zap size={20} className="text-primary" />,
                  title: "AI builds it",
                  body: "Our multi-agent system plans, writes, and validates every file in real time.",
                },
                {
                  step: "03",
                  icon: <Eye size={20} className="text-primary" />,
                  title: "Preview & iterate",
                  body: "See a live preview instantly. Keep chatting to refine or add features.",
                },
              ].map((item) => (
                <div
                  key={item.step}
                  className="bg-card border border-border rounded-xl p-6 hover:border-emerald-900/60 transition-colors"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-xs font-mono text-muted-foreground">{item.step}</span>
                    <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                      {item.icon}
                    </div>
                  </div>
                  <h3 className="font-semibold text-foreground mb-2">{item.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{item.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Feature cards */}
        <section className="py-20 px-6 border-t border-border/50">
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Card 1 — Live build log */}
              <div className="bg-card border border-border rounded-xl p-8">
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  Live build log
                </h3>
                <p className="text-muted-foreground text-sm mb-6 leading-relaxed">
                  Watch the AI planner, builder, and validator work in real time.
                  Every file creation and code fix is visible.
                </p>
                <div className="bg-background rounded-lg p-4 border border-border space-y-2 font-mono text-xs">
                  {[
                    { status: "done", text: "Planner — TodoApp with drag-drop" },
                    { status: "done", text: "create_file  src/App.jsx" },
                    { status: "done", text: "create_file  src/TodoList.jsx" },
                    { status: "running", text: "create_file  src/DragItem.jsx" },
                  ].map((line, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span
                        className={
                          line.status === "done"
                            ? "text-primary"
                            : "text-yellow-400 animate-pulse"
                        }
                      >
                        {line.status === "done" ? "✓" : "○"}
                      </span>
                      <span className="text-muted-foreground">{line.text}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Card 2 — Iterate with chat */}
              <div className="bg-card border border-border rounded-xl p-8">
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  Iterate with conversation
                </h3>
                <p className="text-muted-foreground text-sm mb-6 leading-relaxed">
                  Your project has memory. Add features, fix bugs, or change the
                  design — the AI knows what was built before.
                </p>
                <div className="bg-background rounded-lg p-4 border border-border space-y-3 text-sm">
                  <div className="flex gap-2">
                    <span className="text-muted-foreground shrink-0">You</span>
                    <span className="text-foreground">&ldquo;Add a dark mode toggle&rdquo;</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-primary shrink-0">AI</span>
                    <span className="text-muted-foreground">
                      Updated theme system, added toggle to navbar
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-muted-foreground shrink-0">You</span>
                    <span className="text-foreground">&ldquo;Make the cards rounded&rdquo;</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-primary shrink-0">AI</span>
                    <span className="text-muted-foreground">
                      Applied rounded-xl to all card components
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Mini features strip */}
        <section className="py-16 px-6 border-t border-border/50">
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: <Zap size={18} className="text-primary" />,
                  title: "Production-ready output",
                  body: "Component-based architecture, proper state management, responsive Tailwind styling.",
                },
                {
                  icon: <Eye size={18} className="text-primary" />,
                  title: "Real-time preview",
                  body: "Vite-powered live preview. See your app as it's being built, no waiting.",
                },
                {
                  icon: <Download size={18} className="text-primary" />,
                  title: "Export & deploy",
                  body: "Download the full project. Deploy to Vercel, Netlify, or any platform.",
                },
              ].map((item, i) => (
                <div key={i} className="flex flex-col gap-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                    {item.icon}
                  </div>
                  <h4 className="font-semibold text-foreground text-sm">{item.title}</h4>
                  <p className="text-muted-foreground text-sm leading-relaxed">{item.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="py-24 px-6 border-t border-border/50">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4 leading-tight">
              Ready to build?
            </h2>
            <p className="text-muted-foreground mb-8">
              Start with a free prompt. No credit card required.
            </p>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-emerald-600 transition group"
            >
              Start building for free
              <ArrowRight size={16} className="group-hover:translate-x-0.5 transition" />
            </Link>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border/50 py-10 px-6">
          <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
            <span className="font-semibold text-foreground tracking-tight">Buildable</span>
            <nav className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link href="/" className="hover:text-foreground transition">Home</Link>
              <Link href="/chat" className="hover:text-foreground transition">App</Link>
              <Link href="/signin" className="hover:text-foreground transition">Sign In</Link>
              <Link href="/signup" className="hover:text-foreground transition">Sign Up</Link>
            </nav>
          </div>
        </footer>
      </main>
    </div>
  );
}
