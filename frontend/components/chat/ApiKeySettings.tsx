"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { authApi, type UserData } from "@/api";

interface ApiKeySettingsProps {
  userData: UserData | null;
  onUserDataUpdate: (data: UserData) => void;
}

export function ApiKeySettings({ userData, onUserDataUpdate }: ApiKeySettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleOpen = useCallback(() => setIsOpen(true), []);

  useEffect(() => {
    document.addEventListener("open-settings", handleOpen);
    return () => document.removeEventListener("open-settings", handleOpen);
  }, [handleOpen]);

  const handleSave = async () => {
    if (!apiKey.trim() || apiKey.trim().length < 10) {
      setError("API key must be at least 10 characters.");
      return;
    }
    setSaving(true);
    setError("");
    setSuccess("");
    try {
      const updated = await authApi.saveApiKey(apiKey.trim());
      const merged = { ...userData, ...updated, has_openrouter_key: true };
      localStorage.setItem("user_data", JSON.stringify(merged));
      onUserDataUpdate(merged as UserData);
      setApiKey("");
      setSuccess("API key saved successfully.");
    } catch {
      setError("Failed to save API key. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  const handleRemove = async () => {
    setRemoving(true);
    setError("");
    setSuccess("");
    try {
      await authApi.deleteApiKey();
      const merged = { ...userData, has_openrouter_key: false };
      localStorage.setItem("user_data", JSON.stringify(merged));
      onUserDataUpdate(merged as UserData);
      setSuccess("API key removed.");
    } catch {
      setError("Failed to remove API key.");
    } finally {
      setRemoving(false);
    }
  };

  if (!isOpen) {
    return (
      <Button variant="outline" size="sm" onClick={() => setIsOpen(true)}>
        Settings
      </Button>
    );
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/60"
        onClick={() => { setIsOpen(false); setError(""); setSuccess(""); }}
      />
      {/* Dialog */}
      <div className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md rounded-xl border border-border bg-[#0a0f1a] p-6 shadow-xl">
        <h2 className="text-lg font-semibold text-foreground mb-1">OpenRouter API Key</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Your key is encrypted and stored securely.{" "}
          <a
            href="https://openrouter.ai/keys"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline"
          >
            Get a key
          </a>
        </p>

        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm text-muted-foreground">Status:</span>
          {userData?.has_openrouter_key ? (
            <span className="text-sm text-green-400 font-medium">Key saved</span>
          ) : (
            <span className="text-sm text-yellow-400 font-medium">No key set</span>
          )}
        </div>

        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-or-v1-..."
          className="w-full mb-3 px-3 py-2 rounded-lg border border-border bg-secondary text-foreground text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        />

        {error && <p className="text-red-400 text-sm mb-2">{error}</p>}
        {success && <p className="text-green-400 text-sm mb-2">{success}</p>}

        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            <Button size="sm" onClick={handleSave} disabled={saving || !apiKey.trim()}>
              {saving ? "Saving..." : "Save Key"}
            </Button>
            {userData?.has_openrouter_key && (
              <Button variant="outline" size="sm" onClick={handleRemove} disabled={removing}>
                {removing ? "Removing..." : "Remove Key"}
              </Button>
            )}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => { setIsOpen(false); setError(""); setSuccess(""); }}
          >
            Close
          </Button>
        </div>
      </div>
    </>
  );
}
