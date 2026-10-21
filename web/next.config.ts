import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emit .next/standalone: the traced subset of node_modules plus a server.js,
  // which is what the container in the deployment plan copies instead of
  // installing dependencies in the image. 37 MB against ~500 MB of
  // node_modules. It costs nothing at serving time — `next start` keeps working
  // off .next, and every route measured the same — and about ten seconds of
  // build (25 s to 35 s) for the file tracing, paid once per change.
  output: "standalone",
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
