import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: false, // Prevent double-render/reload in dev
  turbopack: {
    root: process.cwd(),
  },
  // Try to fix HMR loop
  allowedDevOrigins: ['localhost:3000', '127.0.0.1:3000', '192.168.1.130:3000'],
};

export default nextConfig;
