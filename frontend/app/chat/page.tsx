"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { authApi, chatApi, type UserData } from "@/api";
import {
  ChatNavbar,
  ChatInputBox,
  StatusBadge,
  PromotionBanner,
} from "@/components/chat";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [modelChoice, setModelChoice] = useState<"gemini" | "claude">("gemini");
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem("auth_token");
    const user = localStorage.getItem("user_data");

    if (!token) {
      router.push("/signin");
      return;
    }

    setIsAuthenticated(true);

    if (user) {
      try {
        const parsed: UserData = JSON.parse(user);
        setUserData(parsed);

        authApi
          .getCurrentUser()
          .then((freshData) => {
            const updatedUser = { ...parsed, ...freshData };
            localStorage.setItem("user_data", JSON.stringify(updatedUser));
            setUserData(updatedUser);
          })
          .catch((err) => {
            console.error("Failed to refresh user data:", err);
          });
      } catch (err) {
        console.warn("Invalid user_data in localStorage:", err);
        localStorage.removeItem("user_data");
      }
    }
  }, [router]);

  const handleSignOut = () => {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("user_data");
    setIsAuthenticated(false);
    setUserData(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    setIsLoading(true);
    setError("");

    try {
      const response = await chatApi.createChat(input.trim(), modelChoice);
      router.push(`/chat/${response.chat_id}`);
    } catch (err) {
      console.error("Error creating chat:", err);
      setError("Failed to create project. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen w-full relative"
      style={{ backgroundColor: "#030712" }}
    >
      {/* Emerald glow */}
      <div
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 75% 55% at 50% 0%, rgba(16,185,129,0.12), transparent 70%)",
        }}
      />

      <ChatNavbar
        isAuthenticated={isAuthenticated}
        userData={userData}
        onSignOut={handleSignOut}
      />

      <div className="relative z-10 min-h-[calc(100vh-57px)] flex flex-col items-center justify-center px-4">
        <StatusBadge />

        <div className="mb-10 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground tracking-tight">
            What do you want to build?
          </h1>
          <p className="text-muted-foreground mt-3 text-base">
            Describe your app and the AI will plan, code, and preview it live.
          </p>
        </div>

        <div className="w-full max-w-2xl">
          <ChatInputBox
            input={input}
            isLoading={isLoading}
            onInputChange={setInput}
            onSubmit={handleSubmit}
          />

          {/* Model selector */}
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Builder model:</span>
            <div className="flex items-center gap-1 rounded-lg border border-border p-1">
              <button
                type="button"
                onClick={() => setModelChoice("gemini")}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  modelChoice === "gemini"
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                ✦ Gemini 2.5 Pro
              </button>
              <button
                type="button"
                onClick={() => setModelChoice("claude")}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  modelChoice === "claude"
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                ◆ Claude Sonnet 4
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm text-center">
              {error}
            </div>
          )}
        </div>

        <PromotionBanner />
      </div>
    </div>
  );
}
