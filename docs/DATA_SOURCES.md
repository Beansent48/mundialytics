# Fuentes de datos gratuitas y fiables

El proyecto no depende de una única fuente. La idea reproducible es convertir cada fuente a dos schemas canónicos:

- `matches`: histórico con resultado.
- `fixtures`: partidos futuros sin resultado.

## Selecciones

### 1. International football results / Mart Jürisoo

- Uso: resultados de selecciones desde 1872.
- Ideal para: ELO propio, modelos de goles, validación histórica de torneos.
- Limitación: no trae eventos detallados de jugador.
- Encaja en: `matches`.

### 2. World Football Elo Ratings

- Uso: ELO internacional como referencia externa.
- Ideal para: comparar tu ELO propio con un benchmark.
- Limitación: mejor usarlo como validación, no como única fuente del pipeline.

### 3. StatsBomb Open Data

- Uso: eventos, lineups, matches y algunos datos 360 para competiciones seleccionadas.
- Ideal para: player props, tiros, faltas, presión, pases, xG reproducible.
- Limitación: cobertura abierta limitada; no cubre todo el calendario.
- Encaja en: `matches`, `lineups`, `player_events`.

### 4. OpenFootball / football.json / worldcup.json

- Uso: fixtures/resultados públicos en JSON/TXT.
- Ideal para: calendarios, simuladores, demos y fixtures futuros cuando estén cargados.
- Limitación: no tiene eventos avanzados.
- Encaja en: `fixtures` y `matches` ligeros.

## Clubes

### 5. Football-Data.co.uk

- Uso: resultados, odds históricas y algunas estadísticas de clubes en CSV/Excel.
- Ideal para: modelos de clubes y backtesting de mercados 1X2, over/under y hándicap.
- Limitación: no es rico para props de jugador.
- Encaja en: `matches` y `odds`.

### 6. Club Elo

- Uso: ratings ELO de clubes europeos.
- Ideal para: features de fuerza de clubes y comparación con ELO propio.
- Limitación: cobertura centrada en clubes europeos.
- Encaja en: `ratings`.

### 7. soccerdata Python

- Uso: librería Python para acceder a fuentes como Club Elo, FBref, Football-Data, Understat y SoFIFA.
- Ideal para: pipeline Python-first.
- Limitación: scraping sujeto a cambios y términos de cada web.

### 8. OpenFootball para clubes

- Uso: fixtures/resultados públicos de ligas y torneos en JSON/TXT.
- Ideal para: generar `fixtures` futuros si el repositorio está actualizado o para demos reproducibles.
- Limitación: no tiene eventos avanzados ni cuotas.

## Cuotas

### 9. Betfair API

- Uso: cuotas/mercados Exchange y Sportsbook según acceso disponible.
- Ideal para: automatizar lectura de mercados actuales.
- Limitación: autenticación, términos, límites, diferencias entre Exchange/Sportsbook y cobertura variable de player props.

### 10. Football-Data.co.uk para odds históricas

- Uso: odds históricas de clubes.
- Ideal para: comparar las probabilidades del modelo con el precio de mercado.
- Limitación: no cubre player props al nivel necesario para Sustituto+.

## Recomendación práctica

### Para selecciones

1. Histórico: international-results.
2. ELO: propio + benchmark World Football Elo.
3. Eventos de jugador: StatsBomb Open Data cuando haya torneos disponibles.
4. Futuros partidos: fixture CSV manual/oficial/OpenFootball.

### Para clubes

1. Histórico + odds: Football-Data.co.uk.
2. ELO: propio + ClubElo.
3. Eventos/player props: StatsBomb Open Data como base; fuente extra más adelante si necesitas cobertura diaria.
4. Futuros partidos: OpenFootball/Football-Data/manual CSV.

## Realidad importante

Con fuentes gratuitas puedes construir un motor serio de resultados, ELO, goles y algunas props. Para **player props diarios completos** tipo Betfair —faltas de jugador, tiros a puerta, Sustituto+— probablemente necesitarás combinar fuentes abiertas con exportación manual/API/licencia, porque la cobertura gratuita de eventos jugador-partido no es completa para todos los clubes y selecciones.
