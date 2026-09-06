"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

// useLayoutEffect warns during SSR; on the server there is nothing to lay out.
const useIsomorphicLayoutEffect =
  typeof window === "undefined" ? useEffect : useLayoutEffect;

/**
 * Entrance animation that cannot hide the page.
 *
 * The naive version renders at opacity 0 and waits for JavaScript to reveal it,
 * which means a slow bundle, a failed hydration or a throttled background tab
 * leaves a marketing page blank — we caught exactly that, frozen at 0.37
 * opacity. So the server markup is fully visible, the hidden state is applied
 * on the client before the first paint, and the reveal happens on intersection.
 * No JS, no observer, or reduced motion: the content is simply there.
 */
export function Reveal({
  children,
  delay = 0,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: "div" | "li" | "article" | "section";
}) {
  const ref = useRef<HTMLElement>(null);
  const [state, setState] = useState<"idle" | "hidden" | "shown">("idle");

  useIsomorphicLayoutEffect(() => {
    if (
      typeof IntersectionObserver === "undefined" ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }
    setState("hidden");
  }, []);

  useEffect(() => {
    if (state !== "hidden") return;
    const el = ref.current;
    if (!el) return;

    const timers: number[] = [];
    const show = () => setState("shown");

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        // A beat so the browser paints the hidden state first, otherwise an
        // element already on screen jumps straight to shown and the transition
        // never runs. setTimeout, not requestAnimationFrame: rAF is suspended
        // in a background tab, which froze this page half-faded at 0.37.
        timers.push(window.setTimeout(show, 32));
      },
      { rootMargin: "0px 0px -60px 0px" },
    );
    observer.observe(el);

    // Last resort. Whatever goes wrong — no intersection ever reported, a
    // detached container, an observer that never fires — the content appears.
    // Nothing on a marketing page is allowed to stay invisible.
    timers.push(window.setTimeout(show, 1600));

    return () => {
      observer.disconnect();
      timers.forEach(window.clearTimeout);
    };
  }, [state]);

  return (
    <Tag
      // @ts-expect-error — one ref type across the small set of allowed tags
      ref={ref}
      data-reveal={state === "idle" ? undefined : state}
      style={delay ? { transitionDelay: `${delay}s` } : undefined}
      className={cn("mv-reveal", className)}
    >
      {children}
    </Tag>
  );
}
