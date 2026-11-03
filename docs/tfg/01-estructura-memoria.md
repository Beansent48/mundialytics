# Estructura de la memoria: secciones y de qué se alimentan

Andamiaje, no redacción. Cada sección dice **qué va dentro**, **qué material la
alimenta** y **qué falta por generar**. El texto lo escribes tú.

Convención: *(listo)* el material existe y solo hay que contarlo; *(generar)* hay
que producir una tabla, figura o medición; *(tuyo)* depende de información que
solo tienes tú.

Peso orientativo con el corte del 20 de octubre: los capítulos 3, 5, 6 y 7 son el
cuerpo del trabajo. El 3 es largo porque ahí está la computación de datos; los
otros tres son la aportación.

---

## 1. Introducción

| Apartado | Contenido | Material |
|---|---|---|
| 1.1 | Motivación personal: Erasmus en Dortmund, convención de estadística deportiva en Brescia | *(tuyo)* |
| 1.2 | Contexto: analítica deportiva y el mercado de apuestas como patrón de calidad | bibliografía |
| 1.3 | Punto de partida: el proyecto previo, con commit y fecha | `00-punto-de-partida.md` *(listo)* |
| 1.4 | Objetivos: uno general y cinco o seis específicos, medibles | *(generar)* |
| 1.5 | Alcance y exclusiones | `00-punto-de-partida.md` |
| 1.6 | Estructura de la memoria | este documento |

**El 1.4 es el apartado crítico con este corte.** Enuncia los objetivos sobre lo
que puedes garantizar —industrializar el sistema, evaluarlo en vivo, monitorizar
su deriva, ejecutar un experimento de mejora con resultado publicable sea cual
sea— y **no** sobre una mejora de RPS que todavía no existe. Si prometes la
mejora en el 1.4, el capítulo 9 tendrá que explicar por qué no está.

## 2. Estado del arte

| Apartado | Contenido |
|---|---|
| 2.1 | Modelos de goles: Poisson independiente, Maher, Dixon-Coles, Poisson bivariante |
| 2.2 | Valoración de fuerza: Elo y variantes en fútbol |
| 2.3 | xG: construcción, usos y límites |
| 2.4 | Eficiencia del mercado de apuestas y el precio de cierre como referencia |
| 2.5 | Evaluación de pronósticos probabilísticos: RPS, log loss, Brier, calibración |
| 2.6 | Buenas prácticas y MLOps: validación temporal, trazabilidad, deriva, reproducibilidad |
| 2.7 | El hueco que ocupa este trabajo |

El 2.6 es el que justifica el corte: si el estado del arte dice que un modelo sin
monitorización ni trazabilidad no es un sistema, el TFG tiene su tesis servida.
*(generar bibliografía)*

## 3. El sistema de partida y su cadena de datos

Capítulo largo y descriptivo. Aquí está «la chicha», y no se reclama como
aportación. Estructúralo como el recorrido del dato, no como un catálogo:

| Apartado | Contenido | Material |
|---|---|---|
| 3.1 | Fuentes: qué da cada proveedor, cobertura y licencia | *(listo)*, anexo B |
| 3.2 | Degradación de proveedores: Understat muerta, FBref y Sofascore con 403, fixturedownload caída, y las respuestas construidas (ESPN como eje, Kaggle, xG desde comentario de texto) | *(listo)* |
| 3.3 | Ingesta y normalización: resolución de nombres de equipo y jugador, alias, deduplicación | *(listo)* |
| 3.4 | La fundación: esquema, claves, validación cruzada entre fuentes | *(listo)*, anexo C |
| 3.5 | Variables: fuerza ataque-defensa, Elo, xG, forma *walk-forward*, nivel de mercado | *(listo)* |
| 3.6 | El motor: doble Poisson por máxima verosimilitud, matriz de marcador, coherencia entre mercados | *(listo)* |
| 3.7 | Capa de mercados y props, con calibración | *(listo)* |
| 3.8 | Capa de competiciones y simulación Monte Carlo | *(listo)* |
| 3.9 | SquadLab: valoración por rol, cartas, draft, minuto a minuto | *(listo)* |
| 3.10 | Producto web, API y operación diaria | `docs/img/` *(listo)* |
| 3.11 | Estado medido en el corte | tabla de `00-punto-de-partida.md` |

El 3.2 es el apartado más interesante del capítulo y el más fácil de contar: es
una historia real de robustez frente a fuentes que se caen.

## 4. Metodología y protocolo de evaluación

| Apartado | Contenido |
|---|---|
| 4.1 | Por qué RPS y log loss, y por qué el acierto no sirve |
| 4.2 | Validación temporal: *walk-forward*, un fold por temporada, nada aleatorio |
| 4.3 | Regla de despliegue: baseline intocable, experimentos en guiones aparte o tras banderas, se despliega solo lo que mejora fuera de muestra en la mayoría de folds |
| 4.4 | Pre-registro: hipótesis y criterio de éxito antes de ejecutar |
| 4.5 | Registro de resultados negativos |
| 4.6 | Reproducibilidad: entorno fijado, tests como verja, CI |
| 4.7 | Amenazas a la validez: fugas, deriva de temporada, cobertura desigual, comparación contra un solo libro |

Ya practicas todo esto; aquí solo hay que enunciarlo como protocolo y citarlo.
El 4.4 es lo que convierte el capítulo 9 en ciencia y no en búsqueda de un
resultado bonito.

## 5. Industrialización

| Apartado | Contenido | Material |
|---|---|---|
| 5.0 | **Catálogo de defectos**: qué tenía de prototipo el sistema de partida, con el síntoma y la causa de cada fallo | tabla en `00-punto-de-partida.md` y las fichas de `registro/` *(se llena solo)* |
| 5.1 | Diagnóstico y corrección del arranque en frío, con medición antes y después | `docs/COLD_START_DIAGNOSIS.md` *(listo)* |
| 5.2 | Capa de cuentas: modelo de datos, autenticación, aislamiento por usuario, borrado en cascada | `docs/USER_ACCOUNTS_DESIGN.md` *(listo)* |
| 5.3 | Contenerización: composición reproducible de API, web y base de datos | *(por hacer)* |
| 5.4 | Despliegue: VPS, dominio, TLS, secretos, copias de seguridad, despliegue repetible | *(por hacer)* |
| 5.5 | Integración continua y la suite de tests como verja | *(generar cifras en checkout limpio)* |

Encuádralo como ingeniería de datos y MLOps: de prototipo de un usuario en un
portátil a servicio multiusuario reproducible y publicado. No como «hacer la web».

El 5.0 es el apartado que **justifica el capítulo entero**, y conviene abrirlo con
él. Sin el catálogo, industrializar parece una decisión de gusto; con él, es la
respuesta a catorce defectos reales, fechados y medidos, de los cuales **ninguno
levantaba una excepción**: todos dejaban el sistema respondiendo con una cifra
creíble. Ese es el argumento, y la tabla está montada en
`00-punto-de-partida.md`. Cada defecto nuevo que aparezca mientras trabajas suma
una fila, así que no lo dejes para el final: ve añadiéndolo.

## 6. Monitorización del modelo en producción

| Apartado | Contenido | Material |
|---|---|---|
| 6.1 | Qué se registra de cada predicción: huella de modelo, lambdas, cutoff, provenance | *(listo)* |
| 6.2 | Garantía de cobertura: por qué un hueco silencioso es peor que un fallo ruidoso | *(listo)* |
| 6.3 | Informe de deriva: recuentos por partido y calibración de la línea | *(listo)* |
| 6.4 | Criterio de reajuste y alertas | *(generar)* |
| 6.5 | Incidentes reales y lo que se aprendió de ellos | *(listo)* |

El 6.5 tiene material contado y fechado: la contaminación de plantillas al
rellenar un fichero de producción, la caché que se auto-invalidaba, los diez días
en que los mercados de jugador no se registraron. Un capítulo de incidentes y
guardarraíles se defiende muy bien.

## 7. Evaluación en vivo de la temporada 2026/27

El capítulo con más valor y el que hay que generar entero. *(generar)*

| Apartado | Contenido |
|---|---|
| 7.1 | Diseño: predicciones registradas antes del pitido inicial, sin posibilidad de reajuste posterior |
| 7.2 | RPS y log loss en vivo, **agregados por eras de modelo** |
| 7.3 | Calibración: diagramas de fiabilidad y error esperado de calibración |
| 7.4 | En vivo contra backtest: ¿se sostiene el 0.1998 fuera del periodo de ajuste? |
| 7.5 | Desglose por liga y por mercado |
| 7.6 | Cobertura: cuántos partidos se predijeron de los que se jugaron, y por qué faltó alguno |

Fuente: `data/processed/logs/predictions_log.csv`. La separación por eras es
obligatoria: la huella del modelo cambió al desplegar el nivel de mercado y el
afilado al contraste, y mezclarlas haría la cifra ilegible.

## 8. Intento de mejora del modelo

Un experimento, pre-registrado, con el resultado que salga. *(generar)*

| Apartado | Contenido |
|---|---|
| 8.1 | Hipótesis, variable propuesta y de dónde viene la información |
| 8.2 | Criterio de éxito declarado antes: delta mínima y número de folds |
| 8.3 | Diseño y datos |
| 8.4 | Resultado |
| 8.5 | Veredicto y qué dice sobre el techo informativo |

## 9. Resultados

| Apartado | Contenido | Estado |
|---|---|---|
| 9.1 | Rendimiento del sistema antes y después | *(listo)* |
| 9.2 | Capacidades nuevas: multiusuario, reproducible, publicado | *(listo / por hacer)* |
| 9.3 | Resultado de la evaluación en vivo | *(generar)* |
| 9.4 | Resultado del experimento de mejora | *(generar)* |
| 9.5 | Cumplimiento de los objetivos del 1.4, uno por uno | *(generar)* |

## 10. Discusión

Por qué el techo es informativo y no algorítmico, y qué implica para cualquiera
que quiera batir a un libro afilado. Qué significa que el motor sea un activo
analítico y no una ventaja de apuestas. Limitaciones: un solo libro, cinco ligas,
Poisson independiente como base. Ética, juego responsable y la razón de la
licencia no comercial.

## 11. Conclusiones y trabajo futuro

## Anexos

| Anexo | Contenido | Estado |
|---|---|---|
| A | Registro de trabajo del TFG, generado desde `docs/tfg/registro/` | *(se llena solo si sigues la convención)* |
| B | Inventario de fuentes: cobertura, licencia, estado | *(generar)* |
| C | Esquema de la fundación y de las tablas de cuentas | *(listo)* |
| D | Reproducibilidad: entorno, comandos, suite de tests | *(listo)* |
| E | Capturas del producto | `docs/img/` *(listo)* |
| F | Delimitación verificable del punto de partida | *(listo)* |
| G | Registro histórico de experimentos del proyecto previo, con cifras | *(generar desde el CHANGELOG)* |
| H | **Catálogo de defectos**: uno por fila, con síntoma, causa, corrección, commit y **por qué no saltó ninguna alarma** | tabla en `00-punto-de-partida.md` *(se llena solo)* |

El anexo G merece la pena aunque sea de antecedentes: una tabla con los
experimentos previos y sus veredictos sostiene el argumento del techo informativo
del capítulo 10 sin reclamarlos como trabajo propio del TFG.

El anexo H es el que sostiene la tesis. La columna que importa no es la
corrección, es la última: **por qué no saltó ninguna alarma**. Catorce defectos y
ninguno levantaba una excepción, porque todos dejaban el sistema respondiendo
con una cifra creíble. Eso es lo que justifica todo el capítulo 5, y es un
argumento que casi ningún TFG puede hacer con datos propios.

---

## Formato: reglas que conviene fijar antes de escribir

1. **Toda cifra lleva tres cosas**: dentro o fuera de muestra, folds que mejoran
   sobre el total, y contra qué línea base. Una delta suelta no significa nada.
2. **No mezcles las tres series de RPS** (ver `00-punto-de-partida.md`); declara
   la serie en el pie de cada tabla.
3. **Signos coherentes**: en RPS y log loss, menos es mejor. Si escribes −0.00128,
   aclara en el pie que el signo negativo indica mejora.
4. **Decimales**: cuatro para RPS y log loss, cinco solo cuando la delta lo exija.
   No cambies de precisión entre tablas.
5. **Nombres de modelo**: usa la huella, no «el modelo nuevo».
6. **Idioma**: el repositorio está en inglés y la memoria en español. Decide una
   vez si traduces nombres de fichero y de versión —yo los dejaría en original,
   entre comillas— y sé consistente.
7. **Figuras**: numeradas, con pie que diga la conclusión, no el contenido.
8. **Cifras de tests**: siempre las de un checkout limpio, nunca las locales.
9. **Extensión orientativa**: 60–90 páginas sin anexos, con los capítulos 3 y 5-8
   ocupando más de la mitad.

### Bibliografía de arranque, por verificar y completar

- Maher (1982), modelos de Poisson para resultados de fútbol.
- Dixon y Coles (1997), corrección de dependencia e ineficiencia del mercado.
- Karlis y Ntzoufras (2003), Poisson bivariante en fútbol.
- Epstein (1969), origen del *ranked probability score*.
- Constantinou y Fenton (2012), por qué el RPS es la regla adecuada aquí.
- Hvattum y Arntzen (2010), valoraciones Elo como predictor.

Para el 2.6 necesitarás además referencias de MLOps y de validación temporal en
series; pregunta a tu tutor qué línea prefiere. Comprueba títulos y años antes de
citar.

---

## Lo que me falta de ti

1. **Erasmus en Dortmund**: curso, universidad de destino, asignaturas o proyectos
   relacionados, y si hubo contacto con clubes o con gente del sector.
2. **Convención de Brescia**: nombre exacto del evento, fecha y qué te llevaste.
3. **Datos académicos**: universidad, grado, tutor, y si ya hay anteproyecto
   presentado y con qué título y alcance.
4. **Normativa**: plantilla oficial, extensión exigida, estilo de citas y
   convocatoria de entrega.
5. **Si el tutor ya ha aceptado** que el TFG continúe un proyecto propio previo.
