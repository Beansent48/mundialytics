"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/**
 * One line, picked at random on the client.
 *
 * It has to be the client: choosing on the server would either send every
 * visitor the same "random" line (it is prerendered) or make the whole page
 * dynamic for a joke. Nothing is rendered until the pick is made and the line
 * fades in, so the swap reads as intentional rather than as a glitch. The slot
 * keeps its height either way, so nothing below it jumps.
 */
export function RandomLine({
  lines,
  className,
}: {
  lines: string[];
  className?: string;
}) {
  const [index, setIndex] = useState<number | null>(null);

  useEffect(() => {
    setIndex(Math.floor(Math.random() * lines.length));
  }, [lines.length]);

  return (
    <p
      className={cn(
        "min-h-[3.2em] transition-opacity duration-500",
        index === null ? "opacity-0" : "opacity-100",
        className,
      )}
      // The line is decorative flavour; a screen reader announcing a joke on a
      // 404 before the actual explanation is not doing the visitor a favour.
      aria-hidden={index === null}
    >
      {index === null ? "" : lines[index]}
    </p>
  );
}
