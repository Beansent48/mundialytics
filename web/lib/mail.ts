/**
 * Where an account email goes.
 *
 * There is no mail transport yet, and inventing one here would be a dependency
 * on a paid service the project has decided against (~15 €/month total, one
 * VPS and a domain). So the one transport that exists writes the link to the
 * server log, which is a real, usable verification flow for a single operator
 * on their own machine: the link in the terminal works when clicked.
 *
 * It is a seam, not a stub: match alerts (the roadmap backlog) need a real
 * sender, and when one arrives it replaces the body of `send` and nothing else.
 */
export type AccountEmail = {
  to: string;
  subject: string;
  /** The one thing every account email is: a link to click. */
  url: string;
};

export async function send({ to, subject, url }: AccountEmail): Promise<void> {
  // One line, so it is findable in a dev server's output; `console.info`
  // rather than `log` because this is an operational event, not a debug print.
  console.info(`[mail] to=${to} subject="${subject}" url=${url}`);
}
