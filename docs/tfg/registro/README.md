# Registro de trabajo del TFG

Una ficha por pieza de trabajo, desde el corte (`a9c90c8` / tag `tfg-baseline`,
2026-10-20) en adelante. El anexo A de la memoria se genera desde aquí, así que una ficha que no
se escribe es trabajo que no se podrá defender.

## La regla

Toda pieza de trabajo del TFG deja **dos rastros**, en el mismo PR:

1. Una entrada en `CHANGELOG.md`, como siempre.
2. Una ficha en este directorio, con nombre `AAAA-MM-DD-slug.md`, copiada de
   [`PLANTILLA.md`](PLANTILLA.md).

Una pieza de trabajo es cualquier cosa que acabe en un commit con intención
propia: un experimento, una corrección con medición, un componente nuevo, un
incidente y su guardarraíl. No hace falta ficha para un cambio de redacción ni
para una actualización de datos rutinaria.

## Lo que no es negociable

- **La cifra se obtiene ejecutando.** Nunca de memoria, nunca de una nota previa,
  nunca copiada de otra ficha. Si no se ha ejecutado, la ficha dice «sin medir».
- **Los negativos se registran igual que los positivos**, con la misma extensión
  y el mismo formato. Son el resultado, no un hueco.
- **El criterio de éxito se escribe antes de ejecutar.** Rellenar «hipótesis» y
  «criterio de éxito» después de ver el resultado invalida el experimento, y en
  una defensa se nota.
- **Toda cifra lleva su contexto**: dentro o fuera de muestra, folds que mejoran
  sobre el total, y contra qué línea base.
- **Los commits se citan por hash corto**, para que cualquiera reconstruya la
  pieza.

## Por qué así

El proyecto ya se llevó un susto por fiarse de las notas en lugar de verificar
ejecutando: en la auditoría de septiembre, «esto está roto» resultó falso en diez
de once casos, y aparecieron tres bugs vivos que ninguna nota mencionaba. El
registro existe para que la memoria no herede ese problema.

## Índice

| Fecha | Ficha | Veredicto |
|---|---|---|
| 2026-10-21 | [Arranque en frío](2026-10-21-arranque-en-frio.md) | desplegado |
| 2026-11-03 | [Liquidación de predicciones y su auditoría](2026-11-03-liquidacion-predicciones.md) | desplegado |
| 2026-11-04 | [Siete hallazgos anotados y no corregidos](2026-11-04-hallazgos-sin-arreglar.md) | anotado, sin tocar |
| 2026-11-11 | [Evaluación en vivo 2026/27 por era de modelo](2026-11-11-track-record-eras.md) | material listo |

Mantén la tabla al día: es el borrador del anexo A.
