import {
  albumStats,
  readAchievements,
  readProfileStats,
  readProgress,
} from "@/lib/gamification-data";
import { withUser } from "@/lib/user-route";

export const dynamic = "force-dynamic";

export const GET = withUser(async (userId) => {
  const [progress, achievements, stats, album] = await Promise.all([
    readProgress(userId),
    readAchievements(userId),
    readProfileStats(userId),
    albumStats(userId),
  ]);

  return { progress, achievements, stats, album };
});
