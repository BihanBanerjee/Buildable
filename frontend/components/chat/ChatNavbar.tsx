import { Button } from "@/components/ui/button";
import Link from "next/link";
import type { UserData } from "@/api";
import { ProjectsList } from "./ProjectsList";
import { ApiKeySettings } from "./ApiKeySettings";

interface ChatNavbarProps {
  isAuthenticated: boolean;
  userData: UserData | null;
  onSignOut: () => void;
  onUserDataUpdate?: (data: UserData) => void;
}

export function ChatNavbar({
  isAuthenticated,
  userData,
  onSignOut,
  onUserDataUpdate,
}: ChatNavbarProps) {
  return (
    <nav className="relative z-20 border-b border-border">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <Link href={isAuthenticated ? "/chat" : "/"} className="font-semibold text-foreground tracking-tight">
          Buildable
        </Link>
        <div className="flex items-center gap-2">
          {isAuthenticated ? (
            <>
              <ProjectsList />
              <div className="hidden md:flex items-center px-3 py-1.5 rounded-lg bg-secondary border border-border text-sm">
                <span className="text-muted-foreground">{userData?.email}</span>
              </div>
              <ApiKeySettings userData={userData} onUserDataUpdate={onUserDataUpdate || (() => {})} />
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
