import type { MetadataRoute } from "next";

/**
 * Public, not indexed — the user's call for the first deployment (2026-11-02).
 *
 * The site is readable by anybody with the link and the account layer is
 * optional, but nothing here should turn up in a search result yet: the privacy
 * page, the copy and the odds disclaimers are written for someone who was shown
 * the link, not for a stranger arriving from Google.
 *
 * `SITE_INDEXABLE=1` is the whole switch, and it has a second half in
 * `app/[locale]/layout.tsx` (the `robots` metadata, which a direct link still
 * carries). Read at request time, so flipping it needs a restart and not a
 * rebuild.
 */
export const dynamic = "force-dynamic";

export default function robots(): MetadataRoute.Robots {
  const indexable = process.env.SITE_INDEXABLE === "1";
  return {
    rules: indexable
      ? { userAgent: "*", allow: "/" }
      : { userAgent: "*", disallow: "/" },
  };
}
