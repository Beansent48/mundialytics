# The Next.js app: four stages, two of which the compose file uses.
#
#   deps     node_modules from the lockfile
#   source   deps + the sources            <- the `migrate` service runs from here
#   build    `next build`                     (needs the API reachable, see below)
#   runner   .next/standalone only          <- the `web` service runs from here
#
# Why `standalone`: next.config.ts emits `.next/standalone`, the traced subset of
# node_modules plus a server.js -- 37 MB against ~500 MB of node_modules (A1
# measured both). The runner stage therefore installs nothing.
#
# **The build needs a ready API.** Every data page is prerendered
# (`generateStaticParams` over the locales), so `next build` renders each one
# against whatever the API answers at build time, and a build against a down API
# bakes the "the model is warming up" box into static HTML --
# `scripts/require-api.mjs` exists to refuse that and runs here too. That is why
# the stack is brought up in two moves: the API first, then this image.
#
# And the second move is NOT `docker compose build web`. The builder reaches the
# API by borrowing the host's network, host networking in a build is a BuildKit
# *entitlement*, and compose has no flag to grant one — it fails with a bare
# "failed to execute bake: exit status 1". The grant is a bake flag:
#
#   docker buildx bake -f docker-compose.yml web --allow=network.host --load
#
# The README and docs/DEPLOYMENT_PLAN.md carry that line in the sequence.
#
# Context is `web/`; nothing outside it is needed.

ARG NODE_VERSION=22

FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci


FROM deps AS source
COPY . .


FROM source AS build
# Where the browser reaches the API, baked into the client bundle at build time.
# A path, not an origin: Caddy serves it on the site's own origin (see
# docker/Caddyfile), so there is no CORS and the API needs no public hostname.
ARG NEXT_PUBLIC_API_URL=/engine
# Where THIS build, and later the server, reaches the API. An origin, because a
# server-side fetch cannot use a relative URL.
ARG API_INTERNAL_URL=http://127.0.0.1:8000
# Present so that importing the auth layer does not throw during the build. No
# connection is made and no page reads a user at build time; the real values are
# the runtime environment's.
ARG DATABASE_URL=postgres://build:build@127.0.0.1:5432/build
ARG BETTER_AUTH_SECRET=build-only-not-a-secret
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    API_INTERNAL_URL=$API_INTERNAL_URL \
    DATABASE_URL=$DATABASE_URL \
    BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET \
    NEXT_TELEMETRY_DISABLED=1
RUN node scripts/require-api.mjs && npx next build


FROM node:${NODE_VERSION}-alpine AS runner
# tzdata because kickoff times are rendered here as well as priced in the API.
RUN apk add --no-cache tzdata
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    TZ=Europe/Madrid
WORKDIR /app

# The traced server, then the static assets it serves (`next build` leaves
# .next/static out of the standalone copy on purpose: it is CDN-able).
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static

USER node
EXPOSE 3000

# Busybox wget, so nothing is installed for the probe. A locale path rather than
# `/`, which is a redirect, and a page the app can always render -- it does not
# need the API to answer.
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=4 \
  CMD wget -q -O /dev/null http://127.0.0.1:3000/en || exit 1

CMD ["node", "server.js"]
