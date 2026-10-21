"use client";

import { Loader2, Mail } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Link, getPathname, useRouter } from "@/i18n/navigation";
import type { Locale } from "@/i18n/routing";
import { signIn, signUp } from "@/lib/auth-client";
import { errorKey } from "@/lib/auth-errors";
import { cn } from "@/lib/utils";

const inputClass =
  "w-full rounded-[10px] border border-border bg-bg px-3.5 py-2.5 text-[0.88rem] " +
  "outline-none placeholder:text-dim focus:border-brand";

const labelClass =
  "block text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim";

/**
 * Sign in, or create an account, with Google or an email address.
 *
 * One component with a `mode` rather than two: the forms differ by one field
 * and the verb on the button, and the two routes that render it (`/sign-in`
 * and `/sign-up`) are both prerendered, so the switch between them is a link
 * and not a piece of state.
 *
 * Google renders only when the server says it is configured. An OAuth button
 * with no client id behind it leads to Google's own error page, which is a
 * worse answer than not offering it.
 */
export function SignInForm({
  mode,
  google,
}: {
  mode: "signIn" | "signUp";
  google: boolean;
}) {
  const t = useTranslations("auth");
  const locale = useLocale() as Locale;
  const router = useRouter();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signingUp = mode === "signUp";

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);

    const result = signingUp
      ? await signUp.email({ name, email, password })
      : await signIn.email({ email, password });

    if (result.error) {
      setError(t(`errors.${errorKey(result.error.code)}`));
      setBusy(false);
      return;
    }

    // Signing up with verification required creates the user but issues no
    // session, so the token — not the user — is what says whether we are in.
    // Without it, say to go and click the link rather than sending someone to
    // an account page that would tell them they are not signed in.
    if (signingUp && !result.data?.token) {
      setNotice(t("checkYourEmail"));
      setBusy(false);
      return;
    }

    // `refresh` as well as `push`: the account page renders on the server and
    // the router cache would otherwise hand back its signed-out version.
    router.push("/account");
    router.refresh();
  }

  async function onGoogle() {
    setError(null);
    setBusy(true);
    // The callback is computed for the locale the visitor is reading in, so a
    // Spanish visitor comes back to /es/account rather than to English.
    const result = await signIn.social({
      provider: "google",
      callbackURL: getPathname({ href: "/account", locale }),
    });
    if (result.error) {
      setError(t(`errors.${errorKey(result.error.code)}`));
      setBusy(false);
    }
    // On success the browser is already on its way to Google; leave `busy` set.
  }

  return (
    <div className="mx-auto w-full max-w-sm">
      <h1 className="font-display text-[1.9rem] leading-tight">
        {signingUp ? t("signUpTitle") : t("signInTitle")}
      </h1>
      <p className="mt-2.5 text-[0.88rem] leading-relaxed text-muted">
        {t("lead")}
      </p>

      {google ? (
        <>
          <Button
            type="button"
            variant="secondary"
            size="lg"
            onClick={onGoogle}
            disabled={busy}
            className="mt-7 w-full"
          >
            <GoogleMark />
            {t("withGoogle")}
          </Button>
          <div className="my-6 flex items-center gap-3">
            <span className="h-px flex-1 bg-border" />
            <span className="text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim">
              {t("or")}
            </span>
            <span className="h-px flex-1 bg-border" />
          </div>
        </>
      ) : null}

      <form
        onSubmit={onSubmit}
        className={cn("space-y-4", google ? undefined : "mt-7")}
      >
        {signingUp ? (
          <div className="space-y-1.5">
            <label className={labelClass} htmlFor="name">
              {t("name")}
            </label>
            <input
              id="name"
              name="name"
              autoComplete="name"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={inputClass}
            />
          </div>
        ) : null}

        <div className="space-y-1.5">
          <label className={labelClass} htmlFor="email">
            {t("email")}
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="space-y-1.5">
          <label className={labelClass} htmlFor="password">
            {t("password")}
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete={signingUp ? "new-password" : "current-password"}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={inputClass}
          />
          {signingUp ? (
            <p className="text-[0.76rem] text-dim">{t("passwordHint")}</p>
          ) : null}
        </div>

        {error ? (
          <p role="alert" className="text-[0.82rem] text-negative">
            {error}
          </p>
        ) : null}
        {notice ? (
          <p
            role="status"
            className="flex items-start gap-2 text-[0.82rem] text-muted"
          >
            <Mail className="mt-0.5 size-3.5 shrink-0" />
            {notice}
          </p>
        ) : null}

        <Button type="submit" size="lg" disabled={busy} className="w-full">
          {busy ? <Loader2 className="size-4 animate-spin" /> : null}
          {signingUp ? t("createAccount") : t("signIn")}
        </Button>
      </form>

      <p className="mt-6 text-center text-[0.84rem] text-muted">
        {signingUp ? t("haveAccount") : t("noAccount")}{" "}
        <Link
          href={signingUp ? "/sign-in" : "/sign-up"}
          className="font-medium text-brand hover:underline"
        >
          {signingUp ? t("signIn") : t("createAccount")}
        </Link>
      </p>
    </div>
  );
}

/** Google's mark, inlined: four paths beat a network request for an icon. */
function GoogleMark() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden className="size-4">
      <path
        fill="#4285F4"
        d="M45.12 24.5c0-1.56-.14-3.06-.4-4.5H24v8.51h11.84c-.51 2.75-2.06 5.08-4.39 6.64v5.52h7.11c4.16-3.83 6.56-9.47 6.56-16.17z"
      />
      <path
        fill="#34A853"
        d="M24 46c5.94 0 10.92-1.97 14.56-5.33l-7.11-5.52c-1.97 1.32-4.49 2.1-7.45 2.1-5.73 0-10.58-3.87-12.31-9.07H4.34v5.7C7.96 41.07 15.4 46 24 46z"
      />
      <path
        fill="#FBBC05"
        d="M11.69 28.18A13.46 13.46 0 0 1 10.98 24c0-1.45.25-2.86.71-4.18v-5.7H4.34A21.99 21.99 0 0 0 2 24c0 3.55.85 6.91 2.34 9.88l7.35-5.7z"
      />
      <path
        fill="#EA4335"
        d="M24 10.75c3.23 0 6.13 1.11 8.41 3.29l6.31-6.31C34.91 4.18 29.93 2 24 2 15.4 2 7.96 6.93 4.34 14.12l7.35 5.7c1.73-5.2 6.58-9.07 12.31-9.07z"
      />
    </svg>
  );
}
