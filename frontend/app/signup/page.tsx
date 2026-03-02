"use client";

import type React from "react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { authApi } from "@/api";

export default function SignUpPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const data = await authApi.register({ name, email, password });

      if (!data.access_token || !data.user) {
        throw new Error("Invalid response from server");
      }

      localStorage.setItem("auth_token", data.access_token);
      localStorage.setItem("user_data", JSON.stringify(data.user));

      const storedToken = localStorage.getItem("auth_token");
      if (!storedToken) {
        throw new Error("Failed to store authentication token");
      }

      setIsLoading(false);
      router.push("/chat");
    } catch (error: unknown) {
      console.error("Error signing up:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to sign up";

      if (errorMessage.includes("Email already registered")) {
        setError("This email is already registered. Please sign in instead.");
      } else {
        setError(errorMessage);
      }

      setIsLoading(false);
      localStorage.removeItem("auth_token");
      localStorage.removeItem("user_data");
    }
  };

  return (
    <div className="min-h-screen w-full relative" style={{ background: "#030712" }}>
      {/* Emerald radial glow */}
      <div
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 50% at 50% 0%, rgba(16,185,129,0.12), transparent 70%)",
        }}
      />

      <nav className="relative z-20 border-b border-emerald-900/30">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="font-semibold text-lg text-foreground hover:opacity-80 transition">
            Buildable
          </Link>
          <Link href="/signin">
            <Button variant="outline" size="sm">Sign In</Button>
          </Link>
        </div>
      </nav>

      <div className="relative z-10 min-h-[calc(100vh-57px)] flex flex-col items-center justify-center px-4">
        <div className="w-full max-w-md">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-semibold text-foreground mb-2">Create your account</h1>
            <p className="text-muted-foreground text-sm">Start building with AI today</p>
          </div>

          <div className="bg-card border border-border rounded-xl p-8">
            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label htmlFor="name" className="block text-sm text-foreground/80 mb-1.5">
                  Name
                </label>
                <Input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  required
                  disabled={isLoading}
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm text-foreground/80 mb-1.5">
                  Email
                </label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  disabled={isLoading}
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm text-foreground/80 mb-1.5">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  minLength={6}
                  disabled={isLoading}
                />
                <p className="text-xs text-muted-foreground mt-1">Minimum 6 characters</p>
              </div>

              {error && (
                <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                  {error}
                </div>
              )}

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-10 font-medium"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  "Create Account"
                )}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-muted-foreground">
              Already have an account?{" "}
              <Link href="/signin" className="text-primary hover:underline font-medium">
                Sign In
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
