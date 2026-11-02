"use client";

import { Loader2, Trash2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { useRouter } from "@/i18n/navigation";
import { deleteUser } from "@/lib/auth-client";
import { errorKey } from "@/lib/auth-errors";

const inputClass =
  "w-full rounded-[10px] border border-border bg-bg px-3.5 py-2.5 text-[0.88rem] " +
  "outline-none placeholder:text-dim focus:border-brand";

/**
 * Delete the account, for real.
 *
 * Better Auth will only do this for someone who proves it is them: an account
 * with a password has to supply it, and an account that only has Google has to
 * have signed in recently (`session.freshAge`, one day by default). So the form
 * asks for the password when there is one, and when there is not, a stale
 * session comes back as `SESSION_NOT_FRESH` and the message tells the visitor
 * to sign in again — which is a true statement about what is required, not a
 * failure we are hiding.
 *
 * What happens in the database is a cascade, declared in `lib/db/schema.ts`
 * and proved by `tests/account-cascade.test.mjs`, so every row keyed on the
 * user id goes with the user — including the favourites and SquadLab runs that
 * task B2 will add.
 */
export function DeleteAccount({ hasPassword }: { hasPassword: boolean }) {
  const t = useTranslations("auth");
  const router = useRouter();

  const [open, setOpen] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onConfirm(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);

    const result = await deleteUser(hasPassword ? { password } : {});

    if (result.error) {
      setError(t(`errors.${errorKey(result.error.code)}`));
      setBusy(false);
      return;
    }

    router.push("/");
    router.refresh();
  }

  if (!open) {
    return (
      <div>
        <h2 className="text-[0.95rem] font-semibold">{t("deleteTitle")}</h2>
        <p className="mt-1.5 max-w-prose text-[0.85rem] leading-relaxed text-muted">
          {t("deleteBody")}
        </p>
        <Button
          variant="outline"
          size="sm"
          className="mt-4 border-negative text-negative hover:border-negative hover:text-negative"
          onClick={() => setOpen(true)}
        >
          <Trash2 className="size-3.5" />
          {t("deleteAccount")}
        </Button>
      </div>
    );
  }

  return (
    <form onSubmit={onConfirm}>
      <h2 className="text-[0.95rem] font-semibold">{t("deleteConfirmTitle")}</h2>
      <p className="mt-1.5 max-w-prose text-[0.85rem] leading-relaxed text-muted">
        {t("deleteConfirmBody")}
      </p>

      {hasPassword ? (
        <div className="mt-4 max-w-xs space-y-1.5">
          <label
            className="block text-[0.7rem] font-semibold uppercase tracking-[0.11em] text-dim"
            htmlFor="delete-password"
          >
            {t("password")}
          </label>
          <input
            id="delete-password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={inputClass}
          />
        </div>
      ) : null}

      {error ? (
        <p role="alert" className="mt-3 text-[0.82rem] text-negative">
          {error}
        </p>
      ) : null}

      <div className="mt-4 flex flex-wrap gap-2.5">
        <Button
          type="submit"
          size="sm"
          disabled={busy}
          className="bg-negative text-white hover:bg-negative hover:opacity-90"
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
          {t("deleteForever")}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={busy}
          onClick={() => {
            setOpen(false);
            setError(null);
            setPassword("");
          }}
        >
          {t("cancel")}
        </Button>
      </div>
    </form>
  );
}
