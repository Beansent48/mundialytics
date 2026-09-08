import { notFound } from "next/navigation";

/**
 * Catch-all that 404s on purpose.
 *
 * `not-found.tsx` only runs when a segment calls `notFound()`; a URL matching
 * no route at all is a routing miss and Next serves its own bare 404 instead —
 * outside our layout, our theme and our fonts. Routing every unmatched path
 * through here puts the styled page back in charge. A real route always wins
 * over a catch-all, so adding pages needs no change here.
 */
export default function CatchAllNotFound() {
  notFound();
}
