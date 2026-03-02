import type { Metadata } from "next";
import { Geist } from "next/font/google";
import { JetBrains_Mono } from "next/font/google";
import "@fontsource/geist";
import "@fontsource/jetbrains-mono";
import "./globals.css";

const geistSans = Geist({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-geist-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000"
  ),
  title: "Buildable — AI Web App Builder",
  description: "Describe it. Build it. Ship it.",
  icons: {
    icon: "/favicon.ico",
  },
  openGraph: {
    title: "Buildable — AI Web App Builder",
    description:
      "Transform your ideas into production-ready React applications with AI.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Buildable — AI Web App Builder",
    description: "Transform your ideas into production-ready React applications with AI.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${jetbrainsMono.variable} font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
