# v0.46.1 OddsPapi v4 endpoint patch

Fixes RapidAPI 404 errors caused by calling `/sports` instead of the documented `/v4/sports`.

Updated client defaults:

- RapidAPI base URL: `https://odds-api1.p.rapidapi.com`
- Direct base URL: `https://api.oddspapi.io`
- Sports: `/v4/sports`
- Bookmakers: `/v4/bookmakers`
- Markets: `/v4/markets`
- Fixtures: `/v4/fixtures`
- Current odds: `/v4/odds`
- Historical odds: `/v4/historical-odds`

The config file must also use the base URL without `/en` and the endpoint paths with `/v4/...`.
