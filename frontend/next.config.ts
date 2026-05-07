import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the backend URL to be configured via environment variable
  // Set NEXT_PUBLIC_API_URL in Vercel dashboard to your Render backend URL
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000",
  },
};

export default nextConfig;
