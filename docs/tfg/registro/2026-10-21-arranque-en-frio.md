# El arranque en frío deja de ser una carga de página

- **Fecha:** 2026-10-21
- **Tipo:** corrección de rendimiento
- **Versión:** v0.64.0
- **Commits:** `17d6c77`, `6570bd6`, `04ab27a`, `196ce60`, `3d53bfa`, `28d4afc`, `e887d41`, `1c9eedd`, `619be38`, fusionado en `b1796e3`
- **Capítulo de la memoria:** 5.1
- **Veredicto:** desplegado

## Contexto

La primera visita a cualquier página pagaba el ajuste del motor entera: entre 28
y 56 segundos por endpoint, y hasta 115 segundos cuando se cambiaba de motor. Es
el primer obstáculo real para publicar la web, porque un visitante no espera un
minuto.

## Objetivo

Que ninguna visita pague el ajuste del modelo, y que el coste quede en el arranque
del servicio.

## Criterio de éxito

Declarado antes: ningún endpoint por encima de 15 segundos en frío, y el cambio
de motor por debajo de 1 segundo en la primera visita. Medido con cronómetro
dentro del proceso, no a ojo.

## Datos y diseño

Medición instrumentada de cada endpoint en frío y en caliente, antes y después,
sobre el mismo conjunto de datos. El primer commit del lote (`17d6c77`) es
precisamente la medición de partida, para no optimizar a ciegas.

## Resultado

| Caso | Antes | Después |
|---|---|---|
| Endpoint en frío | 28–56 s | 5–14 s |
| Cambio de motor, primera visita | 115 s | 0,06 s |

El camino en caliente son **35 segundos**, de los que `market_level_detail` se
lleva 30,8. Una medición anterior que daba 3,76 segundos era errónea: cronometraba
`joblib.load` y no el camino completo. Queda documentado porque es exactamente el
tipo de cifra que se cuela en una memoria sin que nadie la cuestione.

Cambios que lo consiguen: los modelos se cargan antes de que el servidor se
declare listo; el calendario de liga se sirve desde disco y se refresca por
detrás de la respuesta; memorización del nivel de goles; *single-flight* para que
dos visitas simultáneas no ajusten dos veces; `/health` vuelve a ser una sonda
que no carga nada; y el sitio se ejecuta desde un build de producción, con
negativa a construirlo a ciegas.

## Veredicto

Desplegado. Criterio cumplido en ambas condiciones.

## Qué quedó fuera

Los 30,8 segundos de `market_level_detail` en caliente siguen ahí y son el
siguiente objetivo si hace falta más margen.

## Trampas encontradas

Dos bugs silenciosos que no se buscaban:

1. La caché de `/fixtures/day` **se auto-invalidaba**, así que no cacheaba nada.
2. La temporada se escribía con `current_season` y se leía por fecha. Las dos
   formas coinciden casi todo el año y **divergen en julio**, que es justo cuando
   nadie mira.

Las dos son casos de libro para el capítulo de incidentes: fallos que no se
manifiestan como error, sino como lentitud o como un dato raro fuera de temporada.
