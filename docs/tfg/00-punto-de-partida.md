# Punto de partida del TFG

**Esto no es texto de la memoria.** Es la ficha de contexto: qué existía antes de
empezar el TFG, con qué cifras, y cómo se comprueba.

## El corte

|  |  |
|---|---|
| Commit | `0114c249d7b8d14745c5411f9815c58534fd3ed8` (`0114c24`) |
| Fecha | 2026-10-20 |
| Asunto | *Track record: drift report 2026-10-19* |
| Versión publicada entonces | v0.63.0 |
| Commits previos | 304 de 339 |
| Commits del TFG | 35 a 2026-10-31, y creciendo |

Es el último commit antes de que arranque el trabajo de plataforma: al día
siguiente llega `17d6c77`, *Plan the platform work, and measure what the cold
start costs*. El corte separa limpiamente **el modelo, que queda detrás**, de
**la industrialización, que es el TFG**.

El repositorio arranca el 2026-06-29 ya en la versión v0.50, y el `CHANGELOG.md`
conserva historia hasta v0.49.4: hubo desarrollo anterior incluso sin control de
versiones. El encuadre de «proyecto previo» es demostrable.

### Cómo se verifica

```bash
git log -1 --format='%H %ad %s' --date=short 0114c24
git rev-list --count 0114c24
git rev-list --count 0114c24..HEAD
git show 0114c24:README.md
```

Clava el corte con una etiqueta, para que el tribunal compruebe el límite en
lugar de creérselo:

```bash
git tag -a tfg-baseline 0114c24 -m "Punto de partida del TFG"
```

## La línea base, y el problema que plantea

Cifras del `README.md` en el corte, sobre 10 080 partidos de las cinco grandes
ligas (2020/21–2025/26) contra el cierre de Bet365:

| Métrica | En el corte (20-10-2026) | Hoy |
|---|---|---|
| RPS 1X2 | 0.1998 | 0.1998 |
| Log loss 1X2 | 0.9846 | 0.9846 |
| Log loss O/U 2.5 | 0.6793 | 0.6793 |
| Brecha cubierta | 86 % | 86 % |

Referencias: Bet365 cierra en 0.1946, las tasas base dan 0.2308 y el reparto
uniforme 0.2356. La brecha alcanzable entre las tasas base y el libro es de
**0.0362 de RPS**, y el motor recorre el 85,7 % de ella.

**Las dos columnas son idénticas, y eso es la decisión más importante de este
corte.** El último despliegue de modelo (v0.63.0, el afilado al contraste) cayó
el 19 de octubre, un día antes del corte. Consecuencias:

- El TFG **no hereda** ninguna mejora de métrica: cualquier ganancia de RPS hay
  que producirla dentro del alcance, partiendo de 0.1998.
- El proyecto ya acumula **siete experimentos** que concluyen que la distancia
  que queda hasta el libro es informativa y no algorítmica (diagnóstico de la
  brecha, techo del mercado de totales, dependencia entre goles, forma europea,
  nivel de liga para tiros, valores frescos, señales de motivación y descanso).
  La probabilidad de arrancar RPS significativo en unas semanas es baja.
- Por tanto **la memoria no se puede sostener sobre «he mejorado el RPS»**. Se
  sostiene sobre buenas prácticas, industrialización y evaluación rigurosa, con
  el intento de mejora declarado como experimento cuyo resultado puede ser
  negativo.

Perder ese apoyo no es grave si los objetivos se enuncian bien desde el principio.
Lo que sería grave es escribir el capítulo 1 prometiendo una mejora de métrica y
llegar al capítulo de resultados sin ella.

### Tres series de RPS que no se deben mezclar

- **0.1998** — backtest sobre los 10 080 partidos con cuota de Bet365. Es la
  cifra de portada y la única comparable con el libro.
- **0.2004** — el mismo backtest sobre todos los partidos con predicción,
  incluidos los que no tienen cuota.
- **0.1976** — replay con la señal de alineación, que exige información
  disponible solo una hora antes del partido. No comparable con las otras dos.

Declara en cada tabla cuál usas.

## Los tres pilares de ciencia de datos que el TFG necesita

Con el modelo detrás del corte, hay que garantizar que la memoria no acabe siendo
solo contenedores y autenticación. Estos tres producen material sí o sí:

1. **Evaluación en vivo de la temporada 2026/27.** Las predicciones se registran
   antes del pitido inicial, con huella de modelo por fila, en
   `data/processed/logs/predictions_log.csv`. Es validación fuera de muestra
   genuina y se acumula por sí sola. Hay que agregarla **por eras de modelo**,
   porque la huella cambió al desplegar el nivel de mercado y el afilado al
   contraste. Incluye calibración (diagramas de fiabilidad y error esperado de
   calibración) y comparación con el backtest.
2. **Monitorización del modelo en producción.** Detección de deriva, criterio de
   reajuste, alertas y garantía de cobertura. Es MLOps, y es ciencia de datos
   aplicada, no fontanería: ya existe un informe de deriva diario del que partir.
3. **Un intento de mejora acotado y pre-registrado.** Hipótesis, variable,
   criterio de éxito y número de folds **declarados antes** de ejecutar. Con el
   historial del proyecto, el resultado esperado es negativo, y registrado así
   vale como resultado. Es, de hecho, mejor ciencia que un positivo encontrado
   probando veinte cosas.

## Qué queda fuera: el sistema de partida

Se describe en la memoria con todo su peso —ahí está la mayor parte de la
computación de datos— pero no se reclama como trabajo de TFG.

- **Cadena de datos.** football-data.co.uk (2000/01–2025/26); ESPN como eje
  (temporada completa en una petición y plantilla por partido); FBref vía Kaggle;
  ClubElo propio calculado en local; Understat para el xG histórico; y xG propio
  extraído del comentario de texto de ESPN. Con el historial de degradación de
  proveedores detrás: Understat muerta, FBref y Sofascore devolviendo 403,
  fixturedownload caída entera. Resolución de nombres de equipo y jugador,
  normalización y construcción de la fundación.
- **Motor de partido.** Doble Poisson ataque-defensa por liga por máxima
  verosimilitud (Maher 1982 / Dixon-Coles 1997), Elo con reinicio de temporada,
  predictor dedicado de tasa de xG, forma reciente *walk-forward*, nivel de goles
  leído del mercado, afilado al contraste.
- **Capa de mercados.** 1X2, marcador, over/under, BTTS, primera parte, props de
  equipo y de jugador con calibración de Platt.
- **Capa de competiciones.** Pronóstico de liga desde el punto actual y Champions,
  Europa League y Conference con formato suizo.
- **SquadLab.** Valoración por rol, cartas, draft, Champions histórica y partido
  simulado minuto a minuto.
- **Medición honesta.** Registro pre-partido, provenance por predicción, garantía
  de cobertura y trazabilidad de modelo (v0.53.0 y v0.62.0).

## Qué entra: la aportación del TFG

Hecho a 31 de octubre de 2026:

1. **Arranque en frío** (21-oct, v0.64.0). De 28-56 s a 5-14 s por endpoint, y un
   cambio de motor de 115 s en la primera visita a 0.06 s. Medido con cronómetro,
   no a mano. Modelos cargados antes de que el servidor se declare listo,
   calendario servido desde disco y refrescado por detrás, `/health` convertido
   en sonda real, sitio ejecutado desde build de producción. Dos bugs silenciosos
   encontrados por el camino: una caché que se auto-invalidaba y una temporada
   que se escribía de un modo y se leía de otro.
2. **Capa de cuentas** (21-23 oct, v0.65.0 y v0.66.0). Postgres con herramienta de
   migración, acceso con Google o con correo, borrado de cuenta que arrastra lo
   que cuelga de ella (probado), filas propias por usuario, favoritos y
   preferencias que siguen a la persona entre dispositivos, partidas de SquadLab
   guardadas y limitador de peticiones.
3. **Licencia** (27-oct, v0.66.1). PolyForm Noncommercial 1.0.0.

Por hacer:

4. **Contenerización completa** y composición reproducible de todo el sistema.
5. **Publicación de la web** en un VPS, con lo que eso arrastra: dominio, TLS,
   copias de seguridad, secretos, despliegue repetible.
6. **Los tres pilares** de la sección anterior.

## Volumen, con honestidad

No uses el recuento de líneas: los diffs están dominados por CSV de datos y
registros de predicciones. Usa commits, versiones publicadas (v0.64.0 → v0.66.1 y
las que vengan) y las fichas del registro de trabajo en `docs/tfg/registro/`.
