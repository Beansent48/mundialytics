# Ampliación de líneas de goles en el pipeline de predicción

- **Fecha:** 2026-11-24
- **Tipo:** componente
- **Commits:** `1d61d8a`
- **Capítulo de la memoria:** 5
- **Veredicto:** desplegado

## Contexto

El pipeline de predicción (`log_upcoming_round.py`) solo registraba una línea
de goles por partido: Over/Under 2.5 total, hardcodeada. Sin embargo, el módulo
de líneas dinámicas (`dynamic_lines.py`) ya generaba múltiples líneas para
córners, tarjetas y tiros, con distintos ámbitos (Total, Local, Visitante).
Esta asimetría impedía evaluar el modelo de goles en el mismo nivel de detalle
que los demás mercados de eventos.

## Objetivo

Registrar todas las líneas de goles que el motor es capaz de producir, tanto
totales como por equipo, para poder evaluar la calibración del modelo en cada
línea y ámbito de forma sistemática.

## Criterio de éxito

El prediction log debe contener, para cada partido:
- 6 líneas de goles totales (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
- 4 líneas por equipo local (0.5, 1.5, 2.5, 3.5)
- 4 líneas por equipo visitante (0.5, 1.5, 2.5, 3.5)

Total: 14 filas de mercado "Goles" por partido, frente a 1 anterior.

## Datos y diseño

Las probabilidades de goles totales se calculan a partir de la distribución de
marcadores (`ScoreDistribution.total_goals_probability`), que ya incorpora la
corrección Dixon-Coles (rho) y el ajuste de contraste. Las probabilidades por
equipo usan `probability_for_count_line` sobre la lambda individual (Poisson
marginal).

Ambas funciones ya existían y estaban validadas; el cambio es solo de logging.

Ficheros modificados:
- `src/mundialytics/statistical_core/dynamic_lines.py` — constantes ampliadas
- `scripts/log_upcoming_round.py` — logging de todas las líneas en ambos motores
  (doméstico y europeo)

## Resultado

Suite de tests: 8/8 relevantes pasan. Verificación manual con lambdas 1.5/1.2:

| Línea   | Ámbito     | P(Over) |
|---------|------------|---------|
| O0.5    | Total      | 93.3%   |
| O1.5    | Total      | 75.1%   |
| O2.5    | Total      | 50.6%   |
| O3.5    | Total      | 28.6%   |
| O4.5    | Total      | 13.7%   |
| O5.5    | Total      | 5.7%    |
| O0.5    | Local      | 77.7%   |
| O1.5    | Local      | 44.2%   |
| O2.5    | Local      | 19.1%   |
| O3.5    | Local      | 6.6%    |

Las probabilidades son coherentes con las lambdas y monótonamente decrecientes.

## Veredicto

Desplegado. No cambia la calidad del modelo (mismas lambdas y distribución de
marcadores), pero amplía la capacidad de evaluación: ahora se puede medir la
calibración por línea, por ámbito y en conjunto, con el mismo rigor que los
mercados de córners y tarjetas.

## Qué quedó fuera

La evaluación estadística de estas líneas requiere acumular datos de varias
jornadas. Con el volumen actual (~10 partidos/día × 14 líneas/partido), en un
mes habrá ~4.200 observaciones, suficientes para un análisis de calibración
por línea y ámbito.
