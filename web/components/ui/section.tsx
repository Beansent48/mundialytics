import type { ReactNode } from "react";

import { Reveal } from "@/components/ui/reveal";
import { cn } from "@/lib/utils";

export function Section({
  children,
  className,
  id,
}: {
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("px-5 py-20 sm:px-7 sm:py-28", className)}>
      <div className="mx-auto max-w-[1240px]">{children}</div>
    </section>
  );
}

export function SectionHead({
  eyebrow,
  title,
  lead,
  className,
}: {
  eyebrow: string;
  title: string;
  lead?: ReactNode;
  className?: string;
}) {
  return (
    <Reveal className={cn("max-w-2xl", className)}>
      <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-brand">
        {eyebrow}
      </p>
      <h2 className="font-display mt-3 text-[2rem] leading-[1.12] sm:text-[2.6rem]">
        {title}
      </h2>
      {lead ? (
        <p className="mt-4 text-[0.98rem] leading-relaxed text-muted">{lead}</p>
      ) : null}
    </Reveal>
  );
}
