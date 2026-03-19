import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const API_URL: string =
  (process.env.NEXT_PUBLIC_API_URL as string | undefined) ?? "http://localhost:8000";

export type UserData = {
  id?: string;
  email: string;
  name?: string;
  has_openrouter_key?: boolean;
  [key: string]: unknown;
};
