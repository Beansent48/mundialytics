import { eq } from "drizzle-orm";
import { MailWarning, Shield } from "lucide-react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { headers } from "next/headers";

import { DeleteAccount } from "@/components/account/delete-account";
import { PreferencesForm } from "@/components/account/preferences-form";
import { SignOutButton } from "@/components/account/sign-out-button";
import { buttonStyles } from "@/components/ui/button";
import { Link } from "@/i18n/navigation";
import { auth } from "@/lib/auth";
import { getDb } from "@/lib/db/client";
import { account } from "@/lib/db/schema";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "settings" });
  return { title: t("title"), robots: { index: false } };
}

export default async function SettingsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("settings");
  const tAuth = await getTranslations("auth");

  const session = await auth.api.getSession({ headers: await headers() });

  if (!session) {
    return (
      <div className="mx-auto max-w-[680px] px-5 py-14 sm:px-7 sm:py-20">
        <header>
          <h1 className="font-display text-[2rem] leading-tight sm:text-[2.5rem]">
            {t("title")}
          </h1>
          <p className="mt-3 max-w-xl text-[0.92rem] leading-relaxed text-muted">
            {t("lead")}
          </p>
        </header>

        {/* Preferences work without an account — theme and language affect
            this browser immediately. Only persistence is missing. */}
        <section className="mt-12">
          <h2 className="font-display text-[1.2rem] font-semibold">
            {t("preferences")}
          </h2>
          <p className="mt-1.5 text-[0.84rem] text-muted">{t("preferencesLead")}</p>
          <div className="mt-6">
            <PreferencesForm />
          </div>
        </section>

        <hr className="my-10 border-border" />

        <div className="rounded-[14px] border border-border bg-surface px-6 py-8 text-center">
          <h2 className="font-display text-[1.15rem] font-semibold">
            {t("signedOutTitle")}
          </h2>
          <p className="mx-auto mt-2 max-w-sm text-[0.87rem] leading-relaxed text-muted">
            {t("signedOutBody")}
          </p>
          <Link
            href="/sign-in"
            className={cn(buttonStyles("primary", "md"), "mt-5")}
          >
            {t("signIn")}
          </Link>
        </div>
      </div>
    );
  }

  const { user } = session;

  const providers = await getDb()
    .select({ providerId: account.providerId })
    .from(account)
    .where(eq(account.userId, user.id));
  const hasPassword = providers.some((p) => p.providerId === "credential");

  return (
    <div className="mx-auto max-w-[680px] px-5 py-14 sm:px-7 sm:py-20">
      <header className="flex items-center gap-4">
        {user.image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={user.image}
            alt=""
            width={56}
            height={56}
            className="size-14 rounded-full border border-border object-cover"
          />
        ) : (
          <span className="flex size-14 items-center justify-center rounded-full border border-border bg-surface-2 font-display text-[1.4rem] text-muted">
            {user.name.slice(0, 1).toUpperCase()}
          </span>
        )}
        <div className="min-w-0">
          <h1 className="font-display truncate text-[2rem] leading-tight sm:text-[2.5rem]">
            {t("title")}
          </h1>
          <p className="truncate text-[0.87rem] text-muted">{user.email}</p>
        </div>
      </header>

      {!user.emailVerified ? (
        <p className="mt-7 flex items-start gap-2.5 rounded-[12px] border border-border bg-surface px-4 py-3 text-[0.84rem] leading-relaxed text-muted">
          <MailWarning className="mt-0.5 size-4 shrink-0 text-brand" />
          {tAuth("unverified")}
        </p>
      ) : null}

      {/* 1 · Preferences */}
      <section className="mt-12">
        <h2 className="font-display text-[1.2rem] font-semibold">
          {t("preferences")}
        </h2>
        <p className="mt-1.5 text-[0.84rem] text-muted">{t("preferencesLead")}</p>
        <div className="mt-6">
          <PreferencesForm />
        </div>
      </section>

      <hr className="my-10 border-border" />

      {/* 2 · Account info */}
      <section>
        <h2 className="font-display text-[1.2rem] font-semibold">
          {t("accountSection")}
        </h2>
        <dl className="mt-5 divide-y divide-border border-y border-border text-[0.88rem]">
          <div className="flex items-baseline justify-between gap-4 py-3.5">
            <dt className="text-muted">{t("signedInWith")}</dt>
            <dd className="text-right font-medium">
              {providers.length
                ? providers
                    .map((p) =>
                      p.providerId === "credential"
                        ? t("providerEmail")
                        : t("providerGoogle"),
                    )
                    .join(" · ")
                : "—"}
            </dd>
          </div>
          <div className="flex items-baseline justify-between gap-4 py-3.5">
            <dt className="text-muted">{t("memberSince")}</dt>
            <dd className="text-right font-medium">
              {new Intl.DateTimeFormat(locale, { dateStyle: "long" }).format(
                user.createdAt,
              )}
            </dd>
          </div>
        </dl>
        <div className="mt-6">
          <SignOutButton />
        </div>
      </section>

      <hr className="my-10 border-border" />

      {/* 3 · Data & privacy */}
      <section>
        <div className="flex items-center gap-2.5">
          <Shield className="size-5 text-brand" />
          <h2 className="font-display text-[1.2rem] font-semibold">
            {t("dataSection")}
          </h2>
        </div>
        <p className="mt-3 text-[0.87rem] leading-relaxed text-muted">
          {t("dataLead")}
        </p>

        <div className="mt-7 space-y-6">
          <div>
            <h3 className="text-[0.82rem] font-semibold">{t("dataStored")}</h3>
            <p className="mt-1 text-[0.84rem] leading-relaxed text-muted">
              {t("dataStoredBody")}
            </p>
          </div>
          <div>
            <h3 className="text-[0.82rem] font-semibold">{t("dataNotStored")}</h3>
            <p className="mt-1 text-[0.84rem] leading-relaxed text-muted">
              {t("dataNotStoredBody")}
            </p>
          </div>
          <div>
            <h3 className="text-[0.82rem] font-semibold">{t("dataRights")}</h3>
            <p className="mt-1 text-[0.84rem] leading-relaxed text-muted">
              {t("dataRightsBody")}
            </p>
          </div>
          <p className="text-[0.82rem] text-dim">{t("dataContact")}</p>
        </div>
      </section>

      <hr className="my-10 border-border" />

      {/* 4 · Delete account */}
      <section>
        <p className="mb-4 text-[0.63rem] font-semibold uppercase tracking-[0.14em] text-negative">
          {t("dangerZone")}
        </p>
        <DeleteAccount hasPassword={hasPassword} />
      </section>
    </div>
  );
}
