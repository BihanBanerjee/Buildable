import { Button } from "@/components/ui/button";
import Link from "next/link";
import type { UserData } from "@/api";
import { ProjectsList } from "./ProjectsList";

interface ChatNavbarProps {
  isAuthenticated: boolean;
  userData: UserData | null;
  onSignOut: () => void;
}

export function ChatNavbar({
  isAuthenticated,
  userData,
  onSignOut,
}: ChatNavbarProps) {
  return (
    <nav className="relative z-20 border-b border-border">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <Link href="/" className="font-semibold text-foreground tracking-tight">
          Buildable
        </Link>
        <div className="flex items-center gap-2">
          {isAuthenticated ? (
            <>
              <ProjectsList />
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary border border-border text-sm">
                <span className="text-muted-foreground">{userData?.email}</span>
                <span className="text-border">•</span>
                <span className="text-foreground font-medium">
                  {userData?.tokens_remaining} tokens
                </span>
              </div>
              <Button variant="outline" size="sm" onClick={onSignOut}>
                Sign Out
              </Button>
            </>
          ) : (
            <>
              <Link href="/signin">
                <Button variant="outline" size="sm">Sign In</Button>
              </Link>
              <Link href="/signup">
                <Button size="sm">Sign Up</Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
