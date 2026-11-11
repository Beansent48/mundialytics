import { readProgress } from "@/lib/gamification-data";
import { withUser } from "@/lib/user-route";

export const dynamic = "force-dynamic";

export const GET = withUser((userId) => readProgress(userId));
