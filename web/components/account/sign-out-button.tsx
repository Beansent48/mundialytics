"use client";

import { LogOut } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { useRouter } from "@/i18n/navigation";
import { signOut } from "@/lib/auth-client";

export function SignOutButton() {
  const t = useTranslations("auth");
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  return (
    <Button
      variant="secondary"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await signOut();
        // Home rather than this page: /account would render its signed-out
        // state, which reads like something went wrong. `refresh` drops the
        // cached signed-in header on the way out.
        router.push("/");
        router.refresh();
      }}
    >
      <LogOut className="size-4" />
      {t("signOut")}
    </Button>
  );
}
