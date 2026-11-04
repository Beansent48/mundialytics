# Dockerización: el sitio entero como una pila de compose

- **Fecha:** 2026-11-03
- **Tipo:** componente
- **Versión:** v0.67.0
- **Commits:** `9956ed3`, `ba53568`, `8b770c9`, `c31b96c`, `08c0cdf`, `1aa5bab`, `f540046`, `adead20`, `15e5595`
- **Capítulo de la memoria:** 7 (industrialización)
- **Veredicto:** desplegado en local y verificado en contenedor — con una medición pendiente y nombrada (el contenedor de refresco nunca ha corrido un refresco)

## Contexto

El proyecto se ejecutaba como dos procesos arrancados a mano en una máquina
Windows: `uvicorn api.main:app` y `npm --prefix web run prod`. Las tres tareas
anteriores del plan de plataforma dejaron el terreno preparado — A1 el arranque
en frío (v0.64.0), B1 las cuentas sobre Postgres propio (v0.65.0), B2 los datos
de usuario (v0.66.0) — y A2 es la última: que el sitio completo viva en una
pila de contenedores y que el camino a un servidor esté escrito.

El diseño previo está en `docs/DEPLOYMENT_PLAN.md`, escrito el 2026-10-21 por
la sesión de planificación. Esta ficha registra lo construido y, sobre todo,
en qué se equivocaba ese diseño.

## Hipótesis u objetivo

Que `docker compose up` levante el sitio entero —proxy, web, API, base de
cuentas, migraciones y refresco diario— con los datos montados como volumen y
no dentro de la imagen, y que quede documentado cómo llega eso a un VPS.

No hay hipótesis estadística: ninguna probabilidad servida debe moverse. Eso es
la condición, no el resultado esperado.

## Criterio de éxito

Escrito **antes** de empezar, en el encargo de la tarea (sección A2 de
`docs/PLATFORM_TASK_BRIEFS.md`, redactada el 2026-10-21):

1. `docker compose up` levanta `postgres`, `api`, `web` y `caddy`, y el sitio se
   usa de punta a punta por el proxy con las cuentas funcionando.
2. Las imágenes tienen un tamaño justificado **y medido**, no heredado.
3. La imagen no lleva gigabytes de CSV dentro.
4. `pytest tests/` verde; `npm run lint` y `npx tsc --noEmit` limpios.
5. El plan de despliegue refleja lo construido, con los números medidos y lo que
   no funcionó.

Los cinco están cumplidos y comprobados ejecutando. El 1 y el 2 estuvieron «sin
medir» mientras Docker no arrancaba en esta máquina; se midieron el 2026-11-03/04,
y es ahí donde apareció lo que de verdad aporta esta pieza.

## Datos y diseño

Dos piezas de medición, ambas guiones nuevos que quedan en el repositorio:

- **`scripts/measure_serving_bundle.py`** — instala un gancho de auditoría sobre
  el evento `open` **antes** de importar la aplicación (pandas, joblib y numpy
  abren todos a través del `open` de CPython, así que un solo gancho lo ve todo,
  incluidas las lecturas de nivel de módulo), levanta un uvicorn real en un hilo
  y llama a los 17 endpoints que la API declara: `/health`, `/catalogue`,
  `/leagues`, `/competitions`, `/track-record`, `/fixtures/day`,
  `/fixtures/calendar`, `/fixtures/upcoming`, por cada competición `/league`,
  `/awards` y `/fixtures`, las tres UEFA por `/competition/{slug}`, tres partidos
  jugados y tres por jugar, y SquadLab entero incluida una Champions completa.
  No es una muestra: es la totalidad de la superficie servida, y el guion falla
  si alguna ruta declarada se queda sin ejercitar.
- **`scripts/build_serving_bundle.py`** — convierte ese manifiesto en el
  directorio que monta un despliegue, para poder arrancar la pila con **solo**
  los ficheros medidos. Esa es la prueba de suficiencia: un fichero que falte
  es un 500 en la página que lo necesita, no un apaño silencioso.

Tamaños del árbol con `du -sm data/*` sobre el conjunto real de esta máquina.

## Resultado

Medición del paquete de servicio, ejecutada el 2026-11-03 sobre la aplicación
real (fuera de cualquier simulación: es el proceso que sirve):

| | Ficheros | Leído | Tamaño en disco |
|---|---|---|---|
| `data/external` | 240 | 152,0 MB | 2,1 GB |
| `data/processed` | 35 | 71,2 MB | 508,2 MB |
| `data/curated` | 4 | 31,3 KB | 46,8 KB |
| `data/identity` | 1 | 6,2 KB | 7,9 KB |
| `data/reference` | 1 | 533 B | 1,6 KB |
| código (`src/`, `api/`, `web/data`) | 93 | 1,2 MB | — |
| **total leído** | **373** | **224,5 MB** | — |
| **paquete que se monta** | **277** | **194,5 MB** | **de 26,5 GB** |

El paquete es el **0,7%** del árbol. Queda fuera el código (viaja en las
imágenes) y `data/processed/cache/`, porque la clave del joblib ajustado es un
hash del código del motor y la copia de ese código en un contenedor no es
idéntica byte a byte a la de un checkout de Windows: una caché copiada nunca
acertaría. El contenedor paga un ajuste en frío en su primer arranque, que es
para lo que existe el `start_period` del healthcheck.

Suite y comprobaciones estáticas, ejecutadas:

| | Resultado |
|---|---|
| Suite local (con el dataset construido) | 746 pasados, 3 saltados, 262 s |
| Suite en checkout limpio (CI, python 3.12) | 643 pasados, 112 saltados, 57,8 s |
| `npm run lint`, `npx tsc --noEmit` | limpios |
| `docker compose config` sobre la pila entera | parsea |

Tests nuevos: 9. `tests/test_refresh_loop.py` (6) fija la aritmética del horario
del refresco y el contrato de `last_run.json`; `tests/test_cors_origins.py` (3)
fija que «vacío» y «sin definir» no son lo mismo en la lista de CORS.

## Resultado en contenedor (2026-11-03/04)

| | |
|---|---|
| `mundialytics-api` | **692 MB** — 386 MB de dependencias, **5,2 MB de código del proyecto** |
| `mundialytics-refresh` | **1,01 GB** — filtrar soccerdata de la imagen de servicio ahorra **320 MB** |
| `mundialytics-web` | **352 MB** — solo `.next/standalone` |
| `mundialytics-web-tools` | 1,88 GB — el contenedor de un solo uso de las migraciones |
| las dos imágenes de Python | construidas en 97 s |
| arranque en frío, en contenedor | **394,7 s** y **544,1 s** en dos pasadas |
| arranque con cachés calientes | **21 s** a todo-sano (`models ready in 2.0s`) |
| paquete de servicio | 277 fich./194,5 MB → **583/234,7 MB** → ~356 MB con el fichero de tiros |

El sitio entero por el proxy, en un solo origen: `/`, `/es` y las seis páginas de
datos responden 200 con 58-111 KB de contenido real; `/engine/health` y
`/engine/fixtures/day` responden 200; `/robots.txt` devuelve el Disallow; el alta
de cuenta responde 200, crea la fila de usuario y la cookie de sesión resuelve.
El nivel de mercado sale con sus multiplicadores reales por liga (LaLiga
x1,0451, Serie A x1,0546, …) y el *benchmark* publicado con sus filas (mercado
0,1946, motor 0,1998).

## Veredicto

Construido y documentado; **la verificación en contenedor no se ha hecho**, así
que la ficha no puede decir «desplegado». Lo que sí se decidió, con su razón:

- **Los datos son volumen, nunca capa de imagen.** El refresco reescribe la
  fundación a diario, así que una imagen con un CSV dentro habría que
  reconstruirla una vez al día.
- **El `start_period` del healthcheck es de 420 s y no es relleno.** uvicorn
  ejecuta el *lifespan* —donde se cargan los modelos— **antes** de atar el
  socket, así que una sonda durante un ajuste en frío recibe conexión
  rechazada: no un 503, nada con lo que responder. Sin margen, el contenedor se
  marca enfermo, se reinicia, y el reinicio empieza el mismo ajuste otra vez: no
  levanta nunca. 420 s = los 119 s medidos por A1 en esta máquina, por tres para
  dos vCPU lentas, más el calentamiento de payloads.
- **Un solo origen.** Cinco componentes cliente llamaban a la API desde el
  navegador, así que la pregunta abierta del plan era real. Caddy sirve el sitio
  y entrega `/engine/*` a la API: sin CORS, sin nombre público para la API, un
  salto menos que proxearla por un *route handler* de Next.
- **El refresco diario vive en la pila**, por decisión del usuario del
  2026-11-02 y en contra de la recomendación del propio plan.

## Qué quedó fuera

- **La verificación en contenedor.** Una sesión de Claude no puede arrancar
  Docker en esta máquina: `com.docker.service` está en inicio manual y
  `OpenService` da acceso denegado sin elevación. Se intentó: lanzar
  `Docker Desktop.exe` como usuario arranca la interfaz y la distro de WSL, pero
  el motor no sube sin ese servicio. Queda **sin medir**: que las imágenes
  construyan, cuánto pesan, que la pila levante sana, y el recorrido del sitio
  por el proxy con las cuentas. Cada paso es un comando escrito en
  `docs/DEPLOYMENT_PLAN.md` §6.
- **El commit del track record no se mueve al servidor.** El registro de
  predicciones y el informe de deriva son los dos ficheros refrescados que viven
  en git, y el contenedor no tiene checkout ni credenciales. En la pila se
  acumulan en el volumen; meterlos en git sigue siendo trabajo de la máquina que
  tiene el repositorio. Es el único hueco honesto de la opción elegida.
- **El paquete de 194,5 MB sirve, pero no refresca.** `update_season.py`
  reconstruye la fundación desde los CSV crudos de football-data y añade a los
  ficheros de Understat. La primera transferencia a un servidor lleva el paquete
  más las entradas del refresco; dimensionar eso exige correr el refresco una
  vez contra el paquete y leer qué falta.
- **Cinco rutas probadas y descartadas**, con su razón, en la sección «What did
  not work» del plan de despliegue: el montaje de datos en solo lectura, el
  `python:3.12-slim` que el propio plan especificaba, instalar el lock completo
  en la imagen de servicio, una imagen de web que construya sin API viva, y el
  proxy por *route handler*.

## Trampas encontradas

### Los seis defectos que solo aparecieron al levantar contenedores

Ninguno levantaba una excepción. **Todos dejaban el sistema respondiendo con una
cifra creíble**, que es la columna que importa para el capítulo de incidentes.

| # | Defecto | Por qué no saltó ninguna alarma |
|---|---|---|
| 1 | el paquete medido no podía recalcular el nivel de mercado: 41 MB de entradas escondidas tras un memo cuya clave se construye con un `scandir` | `_club_engine` tiene rama de fallo suave: una línea de log («market level unavailable, lambdas left alone») y todos los partidos tasados sin desplazamiento de mercado |
| 2 | `_market_levels` **memoizaba ese fallo** bajo una clave que describe los ficheros ausentes (`..._e3b0c442.json`, el hash de nada) | los arranques posteriores *aciertan* la caché: responden en menos de un segundo, sin recalcular y sin nivel. Un fallo cacheado se lee como un éxito rápido, y convierte un problema transitorio en permanente |
| 3 | `start_period` de 420 s calculado sobre los 119 s del host, contra 394,7 s y 544,1 s reales | en esta máquina sobraban 25 s, así que el contenedor llegaba a sano. El fallo solo aparece en una máquina más lenta, y ahí es un bucle de reinicios |
| 4 | la imagen de la API excluía `web/data/benchmark_vs_bet365.json`, que la API lee | `_benchmark()` es fail-soft por diseño: devuelve `None` y `/track-record` sirve un payload válido con un campo menos |
| 5 | `web/lib/api.ts` tipaba ese campo como **no nulo** | el compilador no pedía guarda, y en desarrollo el fichero está siempre. El fallo solo existe cuando el servidor manda de verdad el nulo que el tipo negaba — y entonces no fue una tarjeta vacía: `next build` murió prerrenderizando |
| 6 | el paquete no llevaba `understat_shots.csv` (121 MB) | `PlayerPropsModel.fit` lo toma como `if shots is None and shots_path is not None and Path(shots_path).exists()`: el modelo se ajusta, carga y tasa, y la información de penaltis simplemente no está |

Los números 2, 5 y 6 son defectos **del producto**, no del empaquetado, y los
tres tienen la misma forma: una caché que guarda un fallo, un tipo que niega un
nulo que el servidor sí manda, y una entrada cuya ausencia se traga un
`exists()`.

### Y el método de medición tenía el mismo punto ciego que el sistema medido

Esto vale más que cualquiera de los seis. El medidor instrumenta el evento
`open`, y el memo del nivel de mercado construye su clave **haciendo `stat`** de
cada CSV de football-data a través de un glob — y un glob es `os.scandir`, no
`open`. Así que la herramienta que existía para descubrir dependencias ocultas
tenía exactamente la misma ceguera que el código que auditaba: las dependencias
consultadas-pero-no-abiertas eran invisibles a las dos.

Corregido registrando también los listados de directorio (`walked` en el
manifiesto, impreso aparte como «listed (not every file opened)»). La sesión de
documentación del TFG midió el mismo paquete **en frío** (439 ficheros / 356 MB,
rama `claude/settlement-audit`) y encontró el defecto 6 por el otro lado: con
los ajustes cargando de caché, los ficheros que los CONSTRUYERON no se abren
nunca. Dos métodos distintos, el mismo agujero — y su total también queda corto
por el mismo motivo del glob.

### Trampas de proceso y de entorno

- **El plan se equivocaba en el tamaño del árbol por un orden de magnitud:**
  decía 2,7 GB y son **26,5 GB**, porque no contaba `data/raw` (23,9 GB en
  17.256 ficheros).
- **El montaje no puede ser de solo lectura**, como el plan asumía. Servir una
  página escribe: los `raw_*.csv` de la UEFA, las cachés de pronóstico y
  goleadores por liga, las instantáneas de calendario y, en frío, el joblib de
  28 MB. Dos mediciones consecutivas dan conjuntos de escritura distintos porque
  esas cachés solo se reescriben cuando están rancias.
- **`requirements-lock.txt` incluye `soccerdata`**, que arrastra Selenium. La
  imagen de servicio se lo llevaba dentro mientras el comentario dos líneas
  arriba afirmaba lo contrario.
- **`docker compose build web` no puede construir la imagen de web.** La red del
  host en una construcción es un *entitlement* de BuildKit, compose no tiene
  bandera para concederlo y falla con `failed to execute bake: exit status 1` a
  secas. La línea es
  `docker buildx bake -f docker-compose.yml web --allow=network.host --load`.
- **`DATA_DIR` tiene que vivir en `.env`, no en la línea de órdenes.** Pasarlo en
  una llamada y no en la siguiente cambia la configuración del servicio, así que
  compose **recrea** el contenedor contra otro directorio sin avisar. Así se
  produjo el defecto 2.
- **Docker Desktop no sigue los junctions de NTFS.** El `data/` enlazado de un
  worktree se monta como un árbol donde `data/external` y `data/raw` existen y no
  se pueden recorrer, así que en esta máquina la copia del paquete de servicio no
  es una optimización: es la única forma montable de los datos.
- **Un *bind mount* conserva el propietario del host**, y el uid 10001 de la
  imagen no posee nada en un host Linux. Docker Desktop lo oculta por completo,
  así que esto se habría descubierto en el servidor.
- **`core.autocrlf=true` está a nivel de sistema** en esta máquina, así que un
  `.sh` commiteado aquí llega a un servidor con el *shebang* terminado en CR y
  falla con «bad interpreter». Resuelto con `.gitattributes`.
- **El nombre del proyecto de compose venía del directorio**, y A1 ya se había
  encontrado con el Postgres de B1 invisible desde el checkout principal. Fijado
  a `mundialytics`.
- **Las colisiones de versión son silenciosas**: dos ramas escribieron
  `version = "0.66.2"` el mismo día con texto idéntico y git auto-fusionó sin
  marcar conflicto. Solo salió por el CHANGELOG, donde los títulos diferían.
