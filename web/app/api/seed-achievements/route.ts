import { NextResponse } from "next/server";

import { seedAchievements } from "@/lib/gamification-data";

export const dynamic = "force-dynamic";

/**
 * Seed the achievement catalogue. Idempotent: safe to call repeatedly.
 * Only works from localhost or with the revalidate secret.
 */
export async function POST(request: Request) {
  const secret = process.env.REVALIDATE_SECRET;
  const url = new URL(request.url);
  const isLocal =
    url.hostname === "localhost" ||
    url.hostname === "127.0.0.1" ||
    url.hostname === "::1";

  if (!isLocal) {
    const token = url.searchParams.get("secret");
    if (!secret || token !== secret) {
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    }
  }

  await seedAchievements("es");
  return NextResponse.json({ ok: true });
}
