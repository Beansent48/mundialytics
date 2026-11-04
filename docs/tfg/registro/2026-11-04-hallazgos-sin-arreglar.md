# Siete hallazgos anotados y no corregidos, de la revisión del arranque en frío

- **Fecha:** 2026-11-04
- **Tipo:** auditoría
- **Versión:** —
- **Commits:** ninguno (es un registro, no un cambio)
- **Capítulo de la memoria:** 6.5 (incidentes), 4.7 (amenazas a la validez) y 11 (trabajo futuro)
- **Veredicto:** anotado, sin tocar

## Contexto

La revisión del PR #29 (arranque en frío, v0.64.0) dio **once hallazgos**: cuatro
se arreglaron —los dos que pidió el usuario, más un problema de ESPN que se
cerró de rebote y la cobertura de tests— y **siete se anotaron sin tocar**. Los
recoge aquí la sesión de documentación, con el mecanismo y el escenario de fallo
que aportó la sesión de rendimiento.

Van en ficha propia y **no** en el catálogo de defectos de
`00-punto-de-partida.md`, porque ese catálogo es de defectos **corregidos** y
estos siguen vivos. Mezclarlos daría una cuenta falsa.

## Por qué se registran sin arreglarlos

Ninguno estaba en el alcance de aquel PR, cinco no se manifiestan con el tráfico
actual, y arreglar en una revisión lo que la revisión no venía a tocar es la
forma más fácil de ampliar el alcance sin medirlo. Quedan escritos con su
escenario para que la decisión de arreglarlos sea del usuario y con datos.

## Los cinco del patrón «no salta ninguna alarma»

Ninguno lanza excepción; todos dejan el sistema contestando.

**1. El 503 de «calentando» se queda cacheado como página.**
`web/lib/api.ts`. Solo la pantalla de jornada reintenta; `request()` lanza con
cualquier estado no-ok, así que una página renderizada en servidor que se topa
con el 503 pinta su rama `catch` y el ISR cachea **eso** durante toda su ventana.
Escenario: `run_update.ps1` llama a `/api/revalidate` justo tras el refresco e
invalida todas las páginas a la vez, y el refresco también mueve
`data_version()`. Varias se regeneran juntas: la primera toma el lock y recarga
modelos, las demás reciben 503 y cachean el aviso hasta una hora.
`require-api.mjs` protege el `next build` de exactamente esto; nada protege la
revalidación.

**2. El memo del nivel de mercado borra los cachés de otros checkouts.**
`api/engine.py:153` borra todo `market_level_*.json` que no sea el suyo.
Escenario: un worktree y el checkout principal comparten
`data/processed/cache` —lo normal al enlazar `data/`— y con huellas distintas por
CRLF cada proceso escribe su clave y borra la del otro, así que los dos
recalculan `market_level_detail` (28,5 s) en cada arranque, expulsándose
indefinidamente. **Es el mismo patrón que ya costó un refit de 119 s con el glob
de `engine_*.joblib`, y lo escribió la misma sesión que acababa de sufrirlo.**

**3. El `.csv.tmp` tiene nombre fijo y la carrera cruza procesos.**
`api/catalogue.py:135`. El `threading.Lock` solo serializa dentro de un proceso.
Escenario: el paso 7m de `update_season` corre `warm_api_caches.py` en su propio
proceso mientras la API está viva y un hilo de refresco toca la misma liga; los
dos abren `laliga_2026-2027.csv.tmp`, intercalan escrituras y `os.replace`
promueve un fichero mitad y mitad. `_read_snapshot` lanza entonces `ParserError`
—capturado, frame vacío, jornada en blanco— o sirve un calendario truncado.

**4. `_models()` lee `data_version()` dos veces.**
`api/main.py:73` llama a `_ready()` y luego a `club_engine()`, cada uno con su
propia lectura del mtime. Escenario: el refresco cambia la foundation entre las
dos; `_ready()` vio el mtime viejo y volvió sin hacer nada, `club_engine()` ve el
nuevo y falla su `lru_cache`, así que esa petición paga el ajuste completo en
línea —5 s en caliente, ~2 min si cambió el código del motor—, que es justo lo
que el 503 existe para evitar. El cliente aborta a los 20 s.

**5. El frame cacheado del calendario se comparte sin copiar.**
`api/catalogue.py:157`. `_read_snapshot` entrega a todos el mismo DataFrame,
mientras `european.load_state` se cambió **en ese mismo PR** para copiar los
suyos por esta razón exacta («a cached state must not inherit the columns the
last caller added to it»). Escenario: hoy nadie lo muta, así que no está roto; el
primero que añada una columna u ordene in-place envenena el objeto para todos los
lectores posteriores de esa liga-temporada hasta que cambie el mtime. La
inconsistencia con `load_state`, cinco ficheros más allá, invita a ello.

## Los dos que son desperdicio, no defecto

**6. Un hilo por petición mientras hay refresco en vuelo.**
`api/catalogue.py:195`. `_LAST_TRY` se escribe solo después de que vuelva
`_fetch_calendar`, así que durante los ~2 s del refresco toda petición de esa
liga sigue viendo `may_try` en true y arranca otro hilo que falla al tomar el
lock. Cinco visitantes concurrentes crean cinco hilos en el mismo segundo, cuatro
sin hacer nada. Inofensivo con este tráfico, desperdicio puro bajo carga.

**7. `_CACHE_LOCKS` crece sin límite.**
`api/main.py:266`. Se indexa igual que `_CACHE` —fechas, mtimes y huellas en la
tupla— y nada borra nunca una entrada. La API corre semanas y cada día añade
claves por día, competición y versión de datos a los dos diccionarios; `_CACHE`
al menos guarda valores que caducan, `_CACHE_LOCKS` guarda locks que no se
tomarán nunca más. Pequeño por entrada, no acotado en el tiempo, invisible hasta
que alguien perfila.

## Qué aportan a la memoria

El 2 y el 5 son los más instructivos, y conviene contarlos juntos:

- El **2** es un defecto que **la misma sesión que acababa de sufrir ese patrón
  volvió a escribir**, con otro glob y el mismo efecto. Arreglar una instancia no
  arregla el patrón.
- El **5** es una incoherencia **dentro del mismo PR**: un fichero se arregló por
  una razón y el de al lado, cinco ficheros más allá, se quedó sin arreglar por
  la misma razón. Nada falla hoy; el defecto es la invitación.

Para el capítulo 4.7, amenazas a la validez, el 1, el 3 y el 4 comparten una
raíz: **el refresco diario y el servicio corren a la vez y se pisan**. El 1
cachea el aviso de calentamiento, el 3 promueve un CSV a medias y el 4 paga un
ajuste en línea. Los tres aparecen solo en la ventana del refresco, que es
exactamente cuando nadie está mirando.

## Trampa de proceso, de la propia coordinación

La sesión de rendimiento me dijo que el contenido de mi PR #36 ya estaba en
master y que mi rama declaraba v0.66.2. Las dos cosas eran falsas, y ella misma
localizó el motivo: **infirió desde un título del CHANGELOG en vez de comprobar
la ascendencia con `git merge-base --is-ancestor`** — el lado local del conflicto
tenía una entrada titulada como mi PR y de ahí saltó a «ya está dentro».

Es el mismo error que esta sesión cometió con los duplicados del log, agrupando
por el **texto** del partido en vez de por el club resuelto, y concluyendo «sin
duplicados» cuando había 25. Dos sesiones distintas, el mismo día, el mismo
fallo: **comprobar la etiqueta en lugar de la cosa**. Para la memoria vale más
que cualquiera de los siete hallazgos de arriba.
