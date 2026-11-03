# Toda predicción con resultado queda comprobada, y una auditoría lo vigila

- **Fecha:** 2026-11-03
- **Tipo:** corrección + componente
- **Versión:** sin publicar
- **Commits:** sin commitear
- **Capítulo de la memoria:** 6 (monitorización) y 7 (evaluación en vivo)
- **Veredicto:** desplegado

## Contexto

El capítulo 7 necesita el track record en vivo de 2026/27 agregado por eras de
modelo. Al calcularlo por primera vez aparecieron **220 de 412 predicciones 1X2
sin comprobar**, lo que convertía la cifra publicada en un número sobre menos
partidos de los que decía cubrir.

## Objetivo

Que ninguna predicción con resultado disponible quede sin comprobar, y que si
eso vuelve a pasar, se entere alguien.

## Criterio de éxito

Declarado antes: cero predicciones sin liquidar cuyo resultado exista en alguna
fuente del repositorio, y un guardarraíl que falle con código distinto de cero
cuando aparezca una.

## Datos y diseño

`data/processed/logs/predictions_log.csv`, temporada 2026-2027, mercado 1X2.
Guiones: `scripts/tfg/score_live_track_record.py` (cálculo) y
`scripts/audit_settlement.py` (guardarraíl).

## Resultado

Tres bugs vivos en la liquidación de producción, los tres silenciosos:

1. **Nombres en dos espacios distintos.** Los resultados europeos vienen en
   nombres de ClubElo (`Bayern`, `Man United`) y el log en nombres de la
   foundation (`bayern munich`, `manchester united`). El cruce directo acertaba
   **0 de 162** filas: *todas* las predicciones europeas llevaban sin
   comprobarse. Resolver el lado del log al espacio de ClubElo liquida las 108
   que tienen resultado. La canonicalización no sirve: solo pone minúsculas, y
   llega a 15.
2. **Partidos reprogramados.** El cruce exigía fecha exacta, y cuatro partidos
   se jugaron un día distinto del que prometía el calendario cuando se registró
   la predicción (toulouse-troyes, paris sg-lyon, nice-rennes, lyon-angers).
   Se liquidan con una ventana de dos días, y solo si el emparejamiento es
   inequívoco.
3. **La foundation va por detrás del calendario.** La liquidación solo leía la
   foundation, que llega uno o dos jornadas tarde, así que **la jornada más
   reciente no se podía comprobar nunca**. Para un registro en vivo eso es un
   sesgo sistemático, no un retraso. Los calendarios de temporada se usan ahora
   como respaldo, sin tocar lo que la foundation ya tiene.
4. **Un partido desplazado liquidaba su 1X2 y nada más.** La ventana de dos días
   copiaba solo los goles, así que un partido jugado un día antes o después se
   quedaba sin descanso ni eventos **aunque la foundation los tuviera completos
   un día más allá**: 14 partidos de 2026/27. Eso se leía como falta de
   cobertura en lugar de como un cruce que se quedaba a medias. La ventana lleva
   ahora todas las columnas de la fuente, y eso comprueba **644 predicciones
   más**: córners 3608 → 3762, tiros 3280 → 3420, amarillas 2952 → 3078,
   descanso 1665 → 1735, puntos de tarjeta 984 → 1026.

Efecto sobre la cobertura:

| | liquidadas | pendientes |
|---|---|---|
| Antes | 192 | 220 |
| + puente de nombres europeos | 300 | 112 |
| + ventana de dos días | 304 | 108 |
| + respaldo del calendario | **309** | 103 |

De los 103 restantes, 102 son partidos aún sin jugar y **uno** es
vallecano-ath bilbao, que el calendario sitúa el 2026-12-09: se registró con
fecha del 10 de octubre y se aplazó. No queda ninguna predicción con resultado
disponible sin comprobar.

Track record en vivo resultante, 1X2 con trazabilidad:

| | n | RPS | IC 95 % | log loss |
|---|---|---|---|---|
| Todo | 309 | **0.1990** | [0.1836, 0.2149] | 0.9434 |
| Ligas domésticas | 201 | 0.2017 | [0.1845, 0.2192] | 0.9547 |
| Europa | 108 | 0.1939 | [0.1637, 0.2267] | 0.9222 |

Contra el backtest publicado (0.1998 sobre 10 080 partidos) y las tasas base de
esta misma muestra (0.2326). **El motor rinde en vivo como dice su backtest.**

Sesgo por resultado, que es el diagnóstico que destapó el déficit de empate:
local z = +0.88, empate z = −1.03, visitante z = +0.05. Todo dentro del ruido,
así que el arreglo del empate de v0.63.0 no se ha roto en vivo. Calibración:
ECE 0.0264.

Refactor: la pregunta «¿hay resultado para este partido?» estaba escrita tres
veces (producción, auditoría y el guion del TFG), y la auditoría discrepaba de
la liquidación que auditaba — daba por huecos los cuatro partidos desplazados.
Ahora hay una sola función, `attach_results`, que usan los tres.

Suite: 705 pasan, 6 se saltan, antes del refactor. Repetida después.

## Veredicto

Desplegado. Criterio cumplido: la auditoría sale con código 0 y «ninguno» en
huecos sin explicar.

## Qué quedó fuera

El reparto por mercado **ya está asegurado**, no solo medido. La auditoría
trabaja en dos niveles, porque el de partido por sí solo es lo que dejó correr el
agujero europeo durante semanas:

1. **Nivel de partido.** ¿Existe un resultado que la liquidación no usó?
2. **Nivel de fila.** Toda fila que *sí* tiene resultado y no se puntuó debe
   nombrar la columna que le falta, y la auditoría falla si alguna no puede.

| Mercado | registradas | puntuables | puntuadas | no puntuables porque |
|---|---|---|---|---|
| 1X2 y Goles | 622 | 519 | 519 | — |
| ht_1x2 y ht_ft | 622 | 347 | 347 | sin marcador al descanso (172) |
| ht_goles | 1866 | 1041 | 1041 | sin marcador al descanso (516) |
| córners | 5019 | 3762 | 3762 | sin `home_corners` (556) |
| tiros | 4402 | 3420 | 3420 | sin `home_shots` (384) |
| amarillas | 4227 | 3078 | 3078 | sin `home_yellow_cards` (546) |
| puntos de tarjeta | 1188 | 1026 | 1026 | sin tarjetas (15) |
| tiros de jugador | 18866 | 10361 | 10361 | sin datos de jugador (6235) |

Déficit cero en los dieciséis mercados. El clasificador replica las propias
ramas de `evaluate_log`, así que lo que allí hace saltar una fila tiene que
nombrarse aquí.

Lo que sigue pendiente es de **fuente de datos, no de liquidación**: 172 de 519
partidos no traen marcador al descanso —108 son eliminatorias europeas, cuyos
CSV de la UEFA solo traen el resultado final, y el resto son europeas de antes de
la provenance o filas liquidadas por el calendario— y a unas 3.200 filas por
mercado de jugador les faltan los datos del futbolista. Están declarados y
vigilados, pero son cobertura que no tenemos. La foundation, en cambio, está
impecable: 447 filas de la temporada y **cero** sin marcador al descanso.

Tampoco está medida la potencia estadística: con 309 partidos el intervalo del
RPS es de ±0.016, y las diferencias entre eras de modelo son de 0.001. **El
registro en vivo no puede distinguir eras**, y conviene decirlo en la memoria
antes de que lo pregunte el tribunal.

## Trampas encontradas

- `_european_results()` envolvía su carga en `except Exception: return
  pd.DataFrame()`. Cuando fallaba, devolvía vacío **en silencio** y las
  predicciones europeas desaparecían del registro sin un solo aviso. El patrón
  sigue siendo no-fatal a propósito (no se debe reventar una petición HTTP por
  esto), pero ahora quien se entera es la auditoría.
- Una auditoría que no comparte la lógica con lo que audita produce falsos
  positivos, y una auditoría con falsos positivos se acaba ignorando.
- El log guarda la fecha que prometía el calendario, no la que se jugó.
