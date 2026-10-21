import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emit .next/standalone: the traced subset of node_modules plus a server.js,
  // which is what the container in the deployment plan copies instead of
  // installing dependencies in the image. It costs nothing here — `next start`
  // keeps working off .next as before, and the measured build time did not move
  // — and it means the image is not invented from scratch later.
  output: "standalone",
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
