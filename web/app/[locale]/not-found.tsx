import { ArrowRight, Home } from "lucide-react";
import { getTranslations } from "next-intl/server";

import { buttonStyles } from "@/components/ui/button";
import { RandomLine } from "@/components/ui/random-line";
import { Link } from "@/i18n/navigation";
import { cn } from "@/lib/utils";

export default async function NotFound() {
  const t = await getTranslations("notFound");
  const jokes = t.raw("jokes") as string[];

  return (
    <div className="relative flex min-h-[72vh] items-center overflow-hidden px-5 sm:px-7">
      <div
        aria-hidden
        className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(90%_70%_at_50%_40%,black,transparent_75%)]"
      />
      <div className="relative mx-auto w-full max-w-xl text-center">
        {/* The number as a scoreline, which is the only way a football product
            should ever write 404. */}
        <p className="font-display text-[5.5rem] leading-none tracking-[-0.04em] text-brand sm:text-[7rem]">
          4<span className="text-text">0</span>4
        </p>

        <h1 className="font-display mt-4 text-[1.7rem] leading-tight sm:text-[2.1rem]">
          {t("title")}
        </h1>

        <RandomLine
          lines={jokes}
          className="mx-auto mt-5 max-w-md text-[0.98rem] italic leading-relaxed text-brand"
        />

        <p className="mx-auto max-w-md text-[0.88rem] leading-relaxed text-muted">
          {t("body")}
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <Link href="/matchday" className={buttonStyles("primary", "lg")}>
            {t("matchday")}
            <ArrowRight className="size-4" />
          </Link>
          <Link
            href="/"
            className={cn(buttonStyles("secondary", "lg"), "gap-2")}
          >
            <Home className="size-4" />
            {t("home")}
          </Link>
        </div>
      </div>
    </div>
  );
}
