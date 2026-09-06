import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

const base =
  "inline-flex items-center justify-center gap-2 rounded-[10px] font-medium " +
  "transition-[background-color,border-color,color,transform,box-shadow] duration-200 " +
  "ease-[var(--ease-out-quint)] active:scale-[0.985] disabled:pointer-events-none " +
  "disabled:opacity-50 whitespace-nowrap";

const variants = {
  primary:
    "bg-brand text-brand-contrast hover:bg-brand-hover shadow-[var(--shadow-sm)] " +
    "hover:shadow-[0_6px_20px_var(--brand-ring)]",
  secondary:
    "bg-surface text-text border border-border hover:border-border-strong " +
    "hover:bg-surface-2",
  ghost: "text-muted hover:text-text hover:bg-surface-2",
  outline:
    "border border-border-strong text-text hover:border-brand hover:text-brand",
} as const;

const sizes = {
  sm: "h-8 px-3 text-[0.8rem]",
  md: "h-10 px-4 text-[0.875rem]",
  lg: "h-12 px-6 text-[0.95rem]",
  icon: "size-9",
} as const;

export type ButtonProps = ComponentProps<"button"> & {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
};

export function Button({
  className,
  variant = "primary",
  size = "md",
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(base, variants[variant], sizes[size], className)}
      {...props}
    />
  );
}

/** Same skin, for a real link — navigation must stay a link, not a button. */
export function buttonStyles(
  variant: keyof typeof variants = "primary",
  size: keyof typeof sizes = "md",
  className?: string,
) {
  return cn(base, variants[variant], sizes[size], className);
}
