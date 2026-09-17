# Mapeo de mercados tipo Betfair

Este proyecto no depende de Betfair para ejecutarse. Usa un CSV genérico de odds con columnas:

```text
match_id,bookmaker,market_type,team,player,line,odds,substitute_plus
```

## Mercados soportados

```text
match_winner
player_shots
player_shots_on_target
player_fouls_committed
player_fouls_drawn
player_yellow_card
player_goals
player_assists
```

## Ejemplos

```text
player_fouls_committed + Federico Valverde + 1+
player_shots_on_target + Lamine Yamal + 1+
player_fouls_drawn + Nico Williams + 2+
```

## Sustituto+

La lógica implementada para mercados de umbral es:

```text
Si el mercado tiene substitute_plus=1, se ajusta la probabilidad con el sustituto proyectado.
```

El cálculo está en:

```text
src/mundialytics/models/substitute_plus.py
```

Advertencia: las reglas reales pueden variar por casa, país, deporte y mercado. En producción hay que guardar la regla exacta del mercado junto a cada cuota.
