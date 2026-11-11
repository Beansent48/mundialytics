# Evaluación en vivo 2026/27 desglosada por era de modelo

- **Fecha:** 2026-11-11
- **Tipo:** componente
- **Capítulo de la memoria:** 7
- **Veredicto:** material listo para el capítulo

## Contexto

El motor registra cada predicción antes del pitido inicial en
`data/processed/logs/predictions_log.csv`, con una huella de modelo
(`model_fp`) que identifica unívocamente el código, los parámetros y los datos
de entrenamiento del fit que la produjo. Al desplegar el nivel de mercado
(2026-10-17) y el afilado al contraste (2026-10-18), la huella cambió, lo que
permite separar el rendimiento antes y después de cada mejora.

## Hipótesis u objetivo

Responder: ¿se sostiene el RPS 0.1998 del backtest (10 080 partidos, walk-forward
in-sample) cuando el modelo predice partidos que no ha visto? Desglosar por era
para que una mejora o un retroceso no se diluya en la media global.

## Criterio de éxito

Escrito antes de ejecutar: RPS global fuera de muestra ≤ 0.2050 (tolerancia del
~2.5 % sobre el backtest). Ninguna era individual con RPS > 0.2200. ECE < 0.04.

## Datos y diseño

- **Fuente**: `predictions_log.csv`, 78 814 filas de la temporada 2026-2027.
- **Con trazabilidad** (huella de modelo): 53 886 filas (desde ~Oct 9).
- **Sin trazabilidad** (pre-v0.53.0): 24 928 filas (Sep 3 – Oct 21). Estas no
  llevan el trío (p_home, p_draw, p_away) y se descartan del análisis
  probabilístico.
- **Liquidación**: se cruzan con la foundation, los resultados UEFA, los
  calendarios ESPN y un margen de ±2 días para partidos reprogramados, todo en
  una sola llamada a `attach_results` (fuente única de verdad).
- **Guion**: `scripts/tfg/score_live_track_record.py` (1X2) y un script
  complementario para O/U, liga y diagrama de fiabilidad.
- **Referencia**: backtest 0.1998 sobre 10 080 partidos con cuota de Bet365
  (walk-forward, 5 folds). Tasas base del backtest: 0.2308.

## Resultado

### Eras de modelo

| Huella | Tipo | Partidos 1X2 | Fechas | Qué representa |
|---|---|---:|---|---|
| (sin huella) | dom. | 210 (descartados) | Sep 3 – Oct 21 | Pre-v0.53.0, sin trío |
| `03c5a85aea75` | dom. | 177 | Oct 9 – Oct 31 | Primera era con trazabilidad |
| `c5fb8baba200` | dom. | 24 | Oct 25 – Nov 2 | Refit intermedio |
| `21414dc0a156` | dom. | 42 | Nov 1 – Nov 8 | Era actual |
| `euro-74e1db98` | euro. | 108 | Oct 13 – Oct 22 | CL/EL/Conf jornadas 1-3 |
| `euro-74867bbf` | euro. | 54 | Nov 3 – Nov 5 | CL/EL/Conf jornada 4 |

Las huellas solo-jugador (`073318eb087d`, `6b9c48f9ba56`, `8221827fc4f6`,
`2f92991a1016`) afectan a mercados de props y no al 1X2 ni al O/U.

La huella cambia cuando cambian los fuentes del motor, los parámetros o el corte
de entrenamiento. La transición de `03c5a85aea75` a `c5fb8baba200` (solapadas
Oct 25) refleja un refit con más datos; la transición a `21414dc0a156` (Nov 1)
es otro refit. Las eras europeas usan un simulador distinto (sobre ClubElo).

### 1X2: RPS y log loss por era (fuera de muestra)

| Era | n | RPS | IC 95 % bootstrap | Log loss |
|---|---:|---:|---|---:|
| `03c5a85aea75` | 177 | 0.2006 | [0.1832, 0.2188] | 0.9655 |
| `c5fb8baba200` | 24 | 0.2099 | [0.1617, 0.2645] | 0.8750 |
| `21414dc0a156` | 42 | 0.1793 | [0.1521, 0.2072] | 0.8888 |
| `euro-74e1db98` | 108 | 0.1939 | [0.1637, 0.2267] | 0.9222 |
| `euro-74867bbf` | 54 | 0.1944 | [0.1507, 0.2427] | 0.9078 |
| **TOTAL** | **405** | **0.1963** | **[0.1830, 0.2104]** | **0.9329** |
| Domésticas | 243 | 0.1978 | [0.1832, 0.2128] | 0.9433 |
| Europa | 162 | 0.1941 | [0.1683, 0.2206] | 0.9174 |

**Comparación con el backtest**: el RPS global OOS (0.1963) es **mejor** que el
backtest in-sample (0.1998), con un intervalo de confianza que lo incluye
cómodamente. Las tasas base de esta muestra (0.2346) son más altas que las del
backtest (0.2308), lo que indica una temporada más impredecible — el motor
mantiene su ventaja a pesar de ello.

### 1X2 por liga

| Liga | n | RPS | Log loss |
|---|---:|---:|---:|
| Europa (CL/EL/Conf) | 162 | 0.1941 | 0.9174 |
| Serie A | 59 | 0.2002 | 0.9231 |
| Premier League | 50 | 0.2171 | 1.0091 |
| LaLiga | 46 | 0.1785 | 0.9321 |
| Bundesliga | 44 | 0.2042 | 0.9745 |
| Ligue 1 | 44 | 0.1867 | 0.8728 |

La Premier League es la liga con peor RPS (0.2171) y la única con log loss > 1.
LaLiga y Ligue 1 están por debajo de 0.19. Las 5 filas sin liga asignada son
partidos de Ligue 1 con fecha reprogramada (toulouse-troyes, paris sg-lyon,
lorient-brest, nice-rennes, lyon-angers); incluidas, Ligue 1 sería n=44 en
total.

### Sesgo por resultado

| Resultado | Predicho | Observado | Diferencia | z |
|---|---:|---:|---:|---:|
| Local | 0.4607 | 0.4765 | +0.0158 | +0.70 |
| Empate | 0.2237 | 0.1975 | −0.0262 | −1.28 |
| Visitante | 0.3155 | 0.3259 | +0.0104 | +0.49 |

Ningún z es significativo (|z| < 1.96). El empate sigue ligeramente
sobreestimado (−2.6 pp, z = −1.28), coherente con el déficit detectado en
v0.63.0 (z = +5.4 en el backtest), pero aquí con signo invertido y dentro de lo
esperado por azar en n = 405.

### Over/Under 2.5 por era (fuera de muestra)

| Era | n | Log loss | Hit rate |
|---|---:|---:|---:|
| `03c5a85aea75` | 177 | 0.6707 | 55.9 % |
| `c5fb8baba200` | 24 | 0.7063 | 50.0 % |
| `21414dc0a156` | 42 | 0.6891 | 54.8 % |
| `euro-74e1db98` | 108 | 0.6459 | 66.7 % |
| `euro-74867bbf` | 54 | 0.6871 | 53.7 % |
| **TOTAL** | **405** | **0.6703** | **58.0 %** |

El log loss de 0.6703 es comparable con el 0.6931 de un modelo al 50 %
(referencia binaria). La era europea 1 destaca con 66.7 % de acierto.

### Calibración 1X2 (diagrama de fiabilidad, 10 bins)

| Bin | n | Predicho | Observado | ECE contrib. |
|---|---:|---:|---:|---:|
| (0, 0.1] | 69 | 0.0612 | 0.0725 | 0.0006 |
| (0.1, 0.2] | 188 | 0.1583 | 0.1702 | 0.0018 |
| (0.2, 0.3] | 440 | 0.2491 | 0.2114 | 0.0137 |
| (0.3, 0.4] | 166 | 0.3483 | 0.3494 | 0.0001 |
| (0.4, 0.5] | 122 | 0.4452 | 0.5328 | 0.0088 |
| (0.5, 0.6] | 102 | 0.5471 | 0.5784 | 0.0026 |
| (0.6, 0.7] | 60 | 0.6489 | 0.6500 | 0.0001 |
| (0.7, 0.8] | 36 | 0.7427 | 0.7222 | 0.0006 |
| (0.8, 0.9] | 24 | 0.8465 | 0.9167 | 0.0014 |
| (0.9, 1.0] | 8 | 0.9349 | 0.7500 | 0.0012 |

**ECE 1X2 = 0.0310**. El bin (0.2, 0.3] aporta la mayor parte del error:
el motor predice 24.9 % y ocurre 21.1 %, coherente con la ligera
sobreestimación del empate (la mayoría de empates caen en ese rango). El bin
(0.9, 1.0] tiene solo 8 observaciones — no es representativo.

### Calibración O/U 2.5 (5 bins)

| Bin | n | Predicho | Observado | ECE contrib. |
|---|---:|---:|---:|---:|
| (0.2, 0.4] | 15 | 0.3801 | 0.5333 | 0.0057 |
| (0.4, 0.6] | 289 | 0.5204 | 0.5467 | 0.0188 |
| (0.6, 0.8] | 95 | 0.6728 | 0.6842 | 0.0027 |
| (0.8, 1.0] | 6 | 0.8508 | 0.8333 | 0.0003 |

**ECE O/U 2.5 = 0.0274**. El grueso de las predicciones cae en (0.4, 0.6],
donde el modelo subestima ligeramente el over (52 % pred. vs 55 % obs.).

## Cobertura

| Concepto | n |
|---|---:|
| Partidos en la foundation 2026-27 | 500 |
| Partidos predichos (cualquier mercado) | 616 |
| Partidos con huella de modelo | 407 |
| 1X2 predichos (total) | 616 |
| 1X2 predichos (con huella) | 406 |
| 1X2 liquidados | 405 |

La cobertura de liquidación es 405/406 = **99.8 %**. El único partido no
liquidado es vallecano vs ath bilbao (2026-10-10), que falta en la foundation.

Los 210 partidos 1X2 sin huella (Sep 3 – Oct 21) se registraron antes de
v0.53.0, que añadió la trazabilidad y el trío de probabilidades. No pueden
evaluarse probabilísticamente y se descartan.

De los 616 partidos totales, 407 tienen huella — una pérdida de 209 partidos de
la primera fase de la temporada. El motor ya predecía, pero el registro no era
completo.

## Veredicto

**El rendimiento fuera de muestra se sostiene.** RPS 0.1963 en 405 partidos
verdaderamente OOS, vs 0.1998 del backtest en 10 080 partidos in-sample
walk-forward. Ambos criterios de éxito se cumplen:

- RPS global (0.1963) < 0.2050 ✓
- Ninguna era > 0.2200 (la peor es `c5fb8baba200` con 0.2099, n = 24) ✓
- ECE 1X2 (0.0310) < 0.04 ✓

La era más reciente (`21414dc0a156`, Nov 1-8, n = 42) muestra el mejor RPS
individual (0.1793), aunque la muestra es pequeña. Las competiciones europeas
rinden 0.1941, ligeramente mejor que las domésticas (0.1978).

La Premier League es la liga más difícil (RPS 0.2171) y LaLiga la más
predecible (0.1785). El sesgo del empate, que era significativo en el backtest
(z = +5.4), aquí es no significativo (z = −1.28, n = 405).

## Qué quedó fuera

- **Sin huella**: 210 partidos de Sep 3 – Oct 21 no pueden puntuarse. Para el
  TFG se reportan como «fase sin trazabilidad».
- **Mercados de eventos** (corners, tiros, faltas, tarjetas, booking points):
  tienen predicciones liquidadas pero no se analizan aquí. Material para una
  sección complementaria del capítulo 7.
- **Mercados de jugador** (goleador, tiros, asistencia, amarilla): ídem.
- **Mercados de medio tiempo** (HT 1X2, HT O/U, HT-FT): ídem.
- **Comparación con cierre de Bet365**: el backtest usa la cuota de cierre como
  benchmark (0.1946). No hay cuota de cierre en vivo para esta muestra.
- **Desglose temporal** (por jornada o semana): no se incluye; el desglose por
  era captura los cambios estructurales que son más relevantes.

## Trampas encontradas

- **Partidos reprogramados**: 5 fixtures de Ligue 1 se jugaron un día distinto
  al registrado. `_fill_near_date` con tolerancia de 2 días los liquida, pero
  el join de liga falla por fecha exacta. Son domésticos (Ligue 1), no un hueco.
- **Python buffering**: el script de análisis extendido se queda congelado en
  apariencia si no se fuerza `flush=True` o `-u`, porque `attach_results` tarda
  ~30 s y el buffer no se vacía.
- **ClubElo intermitente**: `_elo_resolver` depende de la tabla local de ClubElo
  que a veces cuelga al cargar; en una ejecución se necesitó reintentar.
