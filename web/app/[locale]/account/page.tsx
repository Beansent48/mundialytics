import { eq } from "drizzle-orm";
import { MailWarning } from "lucide-react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { headers } from "next/headers";

import { DeleteAccount } from "@/components/account/delete-account";
import { SignOutButton } from "@/components/account/sign-out-button";
import { buttonStyles } from "@/components/ui/button";
import { Link } from "@/i18n/navigation";
import { auth } from "@/lib/auth";
import { getDb } from "@/lib/db/client";
import { account } from "@/lib/db/schema";
import { cn } from "@/lib/utils";

/**
 * Who you are, and the two things you can do about it.
 *
 * The only page on the site that is rendered per request. Everything else is
 * prerendered over the locales and stays that way: an account exists to unlock
 * saving, not to gate reading, so no football page reads a session.
 *
 * Signed out it says so and offers a link, rather than redirecting. A redirect
 * would turn a URL someone has open in a tab into a surprise, and this page is
 * not a gate.
 */
export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "auth" });
  return { title: t("accountTitle"), robots: { index: false } };
}

export default async function AccountPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("auth");

  const session = await auth.api.getSession({ headers: await headers() });

  if (!session) {
    return (
      <div className="mx-auto max-w-sm px-5 py-20 text-center sm:px-7 sm:py-28">
        <h1 className="font-display text-[1.7rem] leading-tight">
          {t("signedOutTitle")}
        </h1>
        <p className="mt-3 text-[0.9rem] leading-relaxed text-muted">
          {t("signedOutBody")}
        </p>
        <Link href="/sign-in" className={cn(buttonStyles("primary", "md"), "mt-7")}>
          {t("signIn")}
        </Link>
      </div>
    );
  }

  const { user } = session;

  // Which ways this person can get in. `credential` is Better Auth's provider
  // id for an email and password, and its presence is what decides whether
  // deleting the account asks for one.
  const providers = await getDb()
    .select({ providerId: account.providerId })
    .from(account)
    .where(eq(account.userId, user.id));
  const hasPassword = providers.some((p) => p.providerId === "credential");

  return (
    <div className="mx-auto max-w-[680px] px-5 py-14 sm:px-7 sm:py-20">
      <header className="flex items-center gap-4">
        {user.image ? (
          // A Google avatar is an arbitrary remote URL, and routing it through
          // next/image would mean whitelisting a host per identity provider —
          // for one 56px image that the provider already serves optimised.
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
          <h1 className="font-display truncate text-[1.7rem] leading-tight">
            {user.name}
          </h1>
          <p className="truncate text-[0.87rem] text-muted">{user.email}</p>
        </div>
      </header>

      {!user.emailVerified ? (
        <p className="mt-7 flex items-start gap-2.5 rounded-[12px] border border-border bg-surface px-4 py-3 text-[0.84rem] leading-relaxed text-muted">
          <MailWarning className="mt-0.5 size-4 shrink-0 text-brand" />
          {t("unverified")}
        </p>
      ) : null}

      <dl className="mt-9 divide-y divide-border border-y border-border text-[0.88rem]">
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

      <p className="mt-9 max-w-prose text-[0.85rem] leading-relaxed text-muted">
        {t("nothingSavedYet")}
      </p>

      <div className="mt-7">
        <SignOutButton />
      </div>

      <hr className="my-10 border-border" />

      <DeleteAccount hasPassword={hasPassword} />
    </div>
  );
}
