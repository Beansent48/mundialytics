import { cn } from "@/lib/utils";

/**
 * The mark: a stylised pitch centre-circle cut by the halfway line, drawn as a
 * probability arc rather than a ball. It scales down to a favicon and stays
 * legible in one colour, which a crest-style logo would not.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      className={cn("size-8", className)}
    >
      <rect width="32" height="32" rx="9" fill="var(--brand)" />
      <path
        d="M16 5.5v21"
        stroke="var(--brand-contrast)"
        strokeOpacity="0.45"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <circle
        cx="16"
        cy="16"
        r="6.6"
        stroke="var(--brand-contrast)"
        strokeOpacity="0.45"
        strokeWidth="1.4"
      />
      {/* the filled arc: the share of the circle the model claims */}
      <path
        d="M16 9.4a6.6 6.6 0 0 1 5.72 9.9L16 16Z"
        fill="var(--brand-contrast)"
      />
    </svg>
  );
}

export function Wordmark({
  className,
  tagline,
}: {
  className?: string;
  tagline?: string;
}) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <LogoMark />
      <span className="flex flex-col leading-none">
        <span className="text-[1.05rem] font-semibold tracking-[-0.03em]">
          Mundia<span className="text-brand">lytics</span>
        </span>
        {tagline ? (
          <span className="mt-1 text-[0.58rem] font-semibold uppercase tracking-[0.16em] text-dim">
            {tagline}
          </span>
        ) : null}
      </span>
    </span>
  );
}
