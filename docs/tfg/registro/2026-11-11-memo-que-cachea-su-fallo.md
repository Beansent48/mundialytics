# Dos defectos silenciosos de A2, sacados de su PR y con test

- **Fecha:** 2026-11-11
- **Tipo:** corrección
- **Versión:** v0.66.4
- **Commits:** `916962b`, `8ab1c28`, `6ab8782`
- **Capítulo de la memoria:** 5 (defectos corregidos) y 7 (tests como verja)
- **Veredicto:** desplegado

## Contexto

Al construir los contenedores de A2 (tarea A2 del roadmap, PR #35) aparecieron
seis defectos que ningún análisis estático había visto. Dos de ellos **no son de
empaquetado sino de producto**, así que no deberían esperar a que se fusione una
PR de infraestructura de 31 ficheros — y el #35 llevaba una semana atrapado,
conflictando cada vez que algo entraba en master.

Esta ficha cubre esos dos, extraídos a una PR propia. Los encontró y los corrigió
la sesión de A2; aquí se trasladan, se les añade el test que les faltaba y se
registran. El resto de hallazgos de A2 van en la ficha de A2.

## Hipótesis u objetivo

Que los dos arreglos lleguen a master sin arrastrar la infraestructura, y que el
primero quede protegido por un test — porque vive dentro de una optimización de
arranque en frío, que es justo el código que alguien aprieta más tarde sin saber
para qué estaba el retorno temprano.

## Criterio de éxito

**Escrito antes de ejecutar**, para el test; los dos arreglos venían ya medidos
por A2 y aquí no se re-miden (se dice abajo cuáles son suyas).

1. El test debe **fallar con el guardarraíl retirado** y pasar con él. Un test
   que pasa en ambos casos no prueba nada.
2. Al fallar, el fichero que escriba debe terminar en `e3b0c442` — la firma
   exacta que produjeron los contenedores. Si no reproduce esa clave, está
   probando otra cosa.
3. La suite completa sigue verde y `npm run lint` y `tsc --noEmit` limpios.

## Datos y diseño

No hay muestra ni ventana temporal: son defectos de comportamiento, no de
modelo.

- `tests/test_market_level_cache.py`, tres casos: que un nivel vacío **no** se
  escriba, que uno real **sí** se siga escribiendo (el guardarraíl no debe
  costar la optimización que protege), y que un fichero vacío heredado de antes
  del arreglo se recompute en vez de servirse.
- El doble se inyecta sustituyendo el módulo en `sys.modules`, no el atributo:
  `_market_levels` importa `market_level_detail` **dentro** de la función, así
  que parchear `api.engine` no basta.

## Resultado

| Comprobación | Resultado |
|---|---|
| Test con el guardarraíl puesto | 3 de 3 pasan |
| Test con el guardarraíl retirado | 1 falla, 2 pasan — el que debe |
| Fichero escrito al fallar | `market_level_d36b8bd9_k70_2_2026-08-08_0_..._e3b0c442.json` |
| Suite completa | *(ver CHANGELOG: medida en esta misma PR)* |

La clave reproducida termina en `e3b0c442` como se exigía, así que el test
reproduce el defecto real y no una aproximación suya.

**Cifras que son de A2, no medidas aquí:** los tres contenedores que acertaron la
caché envenenada y sirvieron sin nivel de mercado en menos de un segundo; y que
`/en/results` reventaba al prerrenderizar, impidiendo construir la imagen de web.

**Comprobado aparte:** el checkout del usuario no tiene ningún fichero
`e3b0c442`; su única entrada de caché son 194 bytes con los cinco niveles
reales. El defecto nunca se materializó fuera de los contenedores.

## Veredicto

Desplegado. El primero es el grave: el nivel de mercado forma parte del modelo
publicado desde el 17-10, así que servir sin él es **divergencia silenciosa de
la línea base desplegada**, que es precisamente lo que la regla de «la línea base
es sagrada» existe para impedir.

## Qué quedó fuera

- **No se re-midió nada de A2.** Las cifras de contenedor son suyas y están
  citadas como tales. Re-medirlas exigía levantar la pila, que es su tarea.
- **Los otros cuatro hallazgos de A2** siguen en el #35; esto no los adelanta.
- **No se auditó el resto de memos del repo** buscando el mismo patrón
  —cachear un resultado vacío bajo una clave que describe la ausencia de sus
  entradas—, y sería lo natural a continuación: `_market_levels` no tiene por
  qué ser el único.

## Trampas encontradas

- **El patrón del defecto es circular y por eso es invisible.** La clave de
  caché incluye una huella de los ficheros de entrada. Si las entradas no se
  pueden leer, la huella es la del vacío, el resultado es vacío, y el vacío se
  guarda bajo la clave del vacío: la siguiente ejecución con el mismo problema
  **acierta**. El fallo se auto-confirma, y cuanto más falla, más rápido
  responde.
- **Un tipo puede mentir y el compilador ayuda a la mentira.** `_benchmark()`
  devuelve `None` a propósito, pero `web/lib/api.ts` lo tipaba no nulo: `tsc`
  daba el visto bueno y nadie ponía la guarda. En desarrollo nunca era nulo
  porque el fichero siempre está. Una anotación de tipo optimista es una
  aserción sin comprobación.
- **El fallo no se manifestó donde vivía.** El tipo está en el cliente web, pero
  reventó la **construcción de la imagen**, porque la página se prerrenderiza.
  Sin prerrenderizado habría sido un error en la cara de un visitante, en
  producción.
- **Parchear el atributo no basta cuando el import es local.** Una hora de test
  inútil hasta caer en que había que sustituir el módulo entero en `sys.modules`.
