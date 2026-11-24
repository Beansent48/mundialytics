# Mundialytics — Plan de despliegue (2026-11-24)

Dos opciones, de lo más barato a lo más completo. La primera **no cuesta nada**
y resuelve lo que más duele hoy: que el pipeline se para si no enciendes el
PC. La segunda sube todo (web + API + pipeline) a un VPS.

La infraestructura Docker **ya está construida** (tarea A2 del roadmap,
2026-11-03). Lo que falta es decidir cuánto subir y dónde.

---

# Opción A: GitHub Actions gratis (solo pipeline + datos)

**Coste: 0 €.** El repo es público, así que los minutos de Actions son
ilimitados.

Exactamente lo que hace PropsAI: el pipeline corre en un cron de GitHub
Actions, los datos persisten en la caché de Actions (10 GB gratis) y los
resultados (prediction log, drift report) se commitean al repo. La API y la
web siguen en local — las arrancas cuando quieras con datos frescos tras un
`git pull`.

## Qué se gana

- **El pipeline no depende de tu PC.** `update_season.py` corre todos los días
  a las 10:00 (hora de Madrid, 08:00 UTC en invierno) en un runner de GitHub.
- **Las predicciones se loguean antes del kick-off** aunque no estés. Eso es lo
  que hace el track record real.
- **El lineup watcher puede correr** en las horas de partidos (como las 6
  capturas de PropsAI).
- **Los datos frescos llegan con `git pull`.** Cuando enciendas el PC, la API
  y la web arrancan sobre datos actualizados.

## Qué NO se gana

- La web y la API siguen en local. No hay URL pública.
- Si alguien quiere ver la web, tu PC tiene que estar encendido.
- No hay cuentas de usuario (eso necesita Postgres, que necesita un servidor).

## Qué cabe en la caché de Actions (medido)

| Directorio | Tamaño | ¿Necesario? |
|---|---|---|
| `data/raw/football_data/` | 41 MB | Sí — la fundación se reconstruye de aquí |
| `data/external/advanced/understat/` | 223 MB | Sí — el xG histórico |
| `data/external/xg/` | 21 MB | Sí |
| `data/external/clubelo/` | 47 MB | Sí — Elo local |
| `data/external/transfermarkt/` | 881 MB | Opcional — squad values |
| `data/external/uefa/` | 1 MB | Sí |
| `data/processed/` (sin logs ni cache) | ~200 MB | Sí |
| **Total sin Transfermarkt** | **~530 MB** | Cabe de sobra en 10 GB |
| **Total con Transfermarkt** | **~1,4 GB** | Cabe de sobra en 10 GB |

Lo que **NO** se cachea: `data/raw/` (24,5 GB de StatsBomb/Wyscout — el
pipeline no lo toca), `data/external/wyscout_2017_18/` (922 MB — estudio
puntual), `data/processed/logs/runs/` (logs de ejecución — se regeneran).

## Cómo funciona

El workflow está en `.github/workflows/daily_refresh.yml`. Resumen:

1. **Checkout + setup Python 3.12** con caché de pip.
2. **Restaurar la caché de datos** (`actions/cache@v4`, clave rolling
   `data-${{ github.run_id }}` + `restore-keys: data-`). ~530 MB.
3. **Bootstrap** (solo la primera vez): descarga 26 temporadas × 5 ligas de
   football-data.co.uk (~40 MB) si la caché está vacía.
4. **`update_season.py`** con `--full` los lunes o por `workflow_dispatch`.
   Exit 2 (paso requerido falló) se trata como warning, no como error.
5. **Commit del track record**: prediction log, drift report, morning pass,
   lineup pass — los mismos ficheros que `run_update.ps1` commitea en local.
6. **Resumen**: imprime el JSON de `last_steps.json` como tabla.

**Notas:**
- El `--full` se ejecuta los lunes (como en `run_update.ps1`).
- La caché excluye Transfermarkt (881 MB, opcional) y Wyscout (922 MB, estudio
  puntual) para caber de sobra en los 10 GB gratis.
- Pasos opcionales (Sofascore 403, Understat muerta, FBref timeout) fallan sin
  romper el run, igual que en local.
- La primera ejecución construye la foundation SIN xG histórico (Understat no se
  puede re-descargar). El modelo funciona (~0,004 RPS peor). Para el xG
  completo, ver `scripts/seed_actions_cache.py`.

## Pasos para activarlo

1. **Sin secretos que configurar.** El pipeline no usa claves (ESPN es público,
   Understat está muerta, Sofascore da 403). Si algún día se añade una fuente
   con clave, va como repository secret.
2. El workflow ya existe en `.github/workflows/daily_refresh.yml`.
3. Push a master (el workflow se activa al llegar al remoto).
4. Ir a la pestaña **Actions** del repo y lanzar un run manual
   (`workflow_dispatch`) para verificar que funciona.
5. Dejar que el cron haga lo suyo — a las 08:00 UTC todos los días.

## En tu PC

Una vez activado, tu rutina diaria cambia:

```powershell
# Antes: encender el PC a las 10:00, dejar que la tarea Windows corra
# Ahora: cuando quieras ver datos frescos
git pull
# Arrancar la API y la web como siempre
```

La tarea programada de Windows se puede desactivar (no borrar):

```powershell
schtasks /Change /TN "MundialyticsDailyUpdate" /DISABLE
```

---

# Opción B: VPS (~8,50 €/mes) — todo en la nube

---

## 0. Estado actual: qué ya existe

| Pieza | Estado | Dónde |
|---|---|---|
| Dockerfiles (API + refresh, Web) | ✅ Construidos y probados | `api/Dockerfile`, `web/Dockerfile` |
| docker-compose.yml (6 servicios) | ✅ Probado localmente | raíz del repo |
| Caddyfile (proxy inverso + TLS) | ✅ | `docker/Caddyfile` |
| refresh_loop.py (cron sin Windows) | ✅ Equivale a run_update.ps1 | `scripts/refresh_loop.py` |
| .env.example (todas las variables) | ✅ Documentado | raíz del repo |
| Backup de Postgres | ✅ Script + instrucción de cron | `docker/backup-postgres.sh` |
| CI que construye las imágenes | ✅ | `.github/workflows/tests.yml` |
| Plan detallado | ✅ 544 líneas | `docs/DEPLOYMENT_PLAN.md` |

Lo que **NO** existe todavía: un VPS alquilado, un dominio, un despliegue real
con la data sembrada y una ejecución del refresh verificada de principio a fin.

---

## 1. Elegir proveedor y plan

### Recomendación: Hetzner Cloud

| Plan | vCPU | RAM | Disco | Precio | Notas |
|---|---|---|---|---|---|
| **CX22** | 2 (shared) | 4 GB | 40 GB | ~4,15 €/mes | Mínimo viable para servir |
| **CX32** | 2 (shared) | 8 GB | 80 GB | ~7,49 €/mes | **Recomendado** — cabe el refresh en paralelo |
| CX42 | 4 (shared) | 16 GB | 160 GB | ~14,49 €/mes | Solo si el cold start es inaceptable |

**Por qué CX32 (8 GB):** el motor tiene un arranque en frío que carga ~200 MB
de modelos en memoria. Con 4 GB el refresh (que también importa pandas +
sklearn) y el API no caben a la vez cómodamente. 8 GB es holgado y sigue
dentro del presupuesto de ~15 €/mes con el dominio.

**El dato gordo:** `data/` en esta máquina ocupa 26,5 GB, pero lo que el
servidor necesita al principio son ~350 MB (el bundle de serving más los inputs
del refresh). El resto se genera solo con la primera ejecución del refresh.
Con los 80 GB del CX32 hay espacio de sobra para que `data/` crezca.

**Alternativa.** Contabo VPS S (4 vCPU, 8 GB, 200 GB, ~6,99 €/mes) tiene más
disco a precio similar, pero las IPs de Contabo están en más listas negras de
rate limiting, lo que puede afectar a los scrapers del refresh (ESPN, etc.).

### Dominio

Cualquier registrador (~1 €/mes). Ejemplo: `mundialytics.com` o
`mundialytics.es`. Una vez apuntado el registro A al VPS, Caddy obtiene y
renueva el certificado TLS automáticamente.

### Resumen de costes

| Concepto | €/mes |
|---|---|
| VPS Hetzner CX32 | 7,49 |
| Dominio (.com) | ~1,00 |
| **Total** | **~8,50** |

---

## 2. Provisionar y asegurar el VPS

### 2.1 Crear el servidor

1. Crear cuenta en [Hetzner Cloud](https://console.hetzner.cloud)
2. Nuevo servidor: **Ubuntu 24.04**, localización **Falkenstein** (más barato),
   plan **CX32**
3. Añadir tu clave SSH pública al crear (o generarla: `ssh-keygen -t ed25519`)
4. Anotar la IP

### 2.2 Primer acceso y seguridad básica

```bash
ssh root@<IP>

# Actualizar
apt update && apt upgrade -y

# Crear usuario no-root
adduser mundialytics
usermod -aG sudo mundialytics

# Copiar la clave SSH al usuario nuevo
mkdir -p /home/mundialytics/.ssh
cp ~/.ssh/authorized_keys /home/mundialytics/.ssh/
chown -R mundialytics:mundialytics /home/mundialytics/.ssh

# Desactivar login con contraseña y root por SSH
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Firewall
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp    # HTTP/3
ufw enable
```

A partir de aquí, todo como `mundialytics`:

```bash
ssh mundialytics@<IP>
```

### 2.3 Instalar Docker

```bash
# Docker oficial (no el de los repos de Ubuntu, que va atrasado)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker mundialytics

# Cerrar sesión y volver a entrar para que el grupo aplique
exit
ssh mundialytics@<IP>

# Verificar
docker compose version
```

---

## 3. Clonar el repo y sembrar los datos

### 3.1 Clonar

```bash
sudo mkdir -p /srv/mundialytics
sudo chown mundialytics:mundialytics /srv/mundialytics
cd /srv/mundialytics

git clone https://github.com/Beansent48/mundialytics.git .
```

### 3.2 Sembrar `data/`

El servidor necesita los datos que git no lleva (están en `.gitignore`). Desde
el PC con Windows, en PowerShell:

```powershell
# Opción A: copiar solo lo necesario (~350 MB, el serving bundle + inputs del refresh)
# Primero generar el bundle:
python scripts/build_serving_bundle.py --to outputs/bundle_data --clean

# Subirlo:
scp -r outputs/bundle_data/* mundialytics@<IP>:/srv/mundialytics/data/
```

```powershell
# Opción B: copiar toda la carpeta data/ (~2.6 GB sin data/raw)
# Más simple, todo funciona de inmediato incluido el refresh
rsync -avz --exclude="raw/" data/ mundialytics@<IP>:/srv/mundialytics/data/
```

**Recomendación: opción B.** La opción A es suficiente para servir pero el
refresh necesita más archivos (Understat, football-data CSVs, etc.) y si no
los encuentra falla en pasos no opcionales. Con la B el primer refresh pasa
limpio.

> **Nota sobre `data/raw/` (23,9 GB):** Son dumps de StatsBomb/Wyscout de
> estudios puntuales. El motor no los lee para servir ni para el refresh.
> No hace falta copiarlos.

### 3.3 Permisos

```bash
# El contenedor corre como el usuario que posee data/
id -u    # → 1000 (típicamente)
id -g    # → 1000
```

Estos valores van en `.env` como `DATA_UID` y `DATA_GID`.

---

## 4. Configurar `.env`

```bash
cd /srv/mundialytics
cp .env.example .env
```

Editar `.env` con los valores de producción:

```env
# --- Datos ---
DATA_DIR=./data
DATA_UID=1000
DATA_GID=1000

# --- Hora ---
TZ=Europe/Madrid

# --- Dirección pública ---
SITE_ADDRESS=mundialytics.example.com    # tu dominio real
ACME_EMAIL=tu@email.com                  # para Let's Encrypt
HTTP_PORT=80
HTTPS_PORT=443

# --- API ---
API_PORT=8000
API_CORS_ORIGINS=                        # vacío = sin CORS (correcto en producción)

# --- Cuentas ---
POSTGRES_USER=mundialytics
POSTGRES_PASSWORD=<contraseña-segura>    # cambia esto
POSTGRES_DB=mundialytics
POSTGRES_PORT=5432
DATABASE_URL=postgres://mundialytics:<contraseña-segura>@postgres:5432/mundialytics

# Generar con:
#   node -e "console.log(require('node:crypto').randomBytes(32).toString('base64url'))"
BETTER_AUTH_SECRET=<secret-generado>
BETTER_AUTH_URL=https://mundialytics.example.com
GOOGLE_CLIENT_ID=                        # opcional
GOOGLE_CLIENT_SECRET=                    # opcional

# --- Refresh diario ---
REFRESH_AT=10:00
REFRESH_FULL_DAY=Monday
REFRESH_ON_START=0
REFRESH_ARGS=

# --- Revalidación ---
# OBLIGATORIO para que el refresh invalide la caché de la web
REVALIDATE_SECRET=<cualquier-string-aleatoria>

# --- SEO ---
SITE_INDEXABLE=                          # vacío = noindex; poner 1 cuando quieras ser visible
```

---

## 5. Levantar la pila

La secuencia es de **dos pasos** porque la imagen web necesita la API lista
para prerenderizar las páginas estáticas.

```bash
cd /srv/mundialytics

# Paso 1: levantar Postgres y la API (espera a que estén healthy)
docker compose up -d --wait postgres api

# Paso 2: construir la imagen web (necesita host networking para llegar a la API)
docker buildx bake -f docker-compose.yml web --allow=network.host --load

# Paso 3: levantar todo
docker compose up -d

# Verificar
docker compose ps
```

**El primer arranque tarda.** La API tiene un cold start de 5-20 minutos en 2
vCPU mientras carga los modelos (el `start_period: 1200s` del healthcheck lo
cubre). Una vez caliente, el reinicio es de segundos.

Verificar que todo está bien:

```bash
# Salud de la API
curl -s http://localhost:8000/health | python3 -m json.tool

# Salud de la web (a través de Caddy)
curl -s -o /dev/null -w "%{http_code}" http://localhost/en

# Logs
docker compose logs -f --tail=50
```

---

## 6. Dominio y TLS

1. Comprar el dominio en tu registrador
2. Crear un registro **A** apuntando a la IP del VPS
3. Esperar propagación DNS (5-30 min)
4. Asegurarse de que `.env` tiene `SITE_ADDRESS=tu-dominio.com` y
   `ACME_EMAIL=tu@email.com`
5. Reiniciar Caddy: `docker compose restart caddy`

Caddy obtiene el certificado Let's Encrypt automáticamente y lo renueva solo.
No hay nada más que hacer para HTTPS.

---

## 7. Backups

### Postgres (cuentas de usuario)

```bash
# Instalar el cron
sudo crontab -u mundialytics -e
```

Añadir:

```cron
30 4 * * *  /srv/mundialytics/docker/backup-postgres.sh >> /var/log/mundialytics-backup.log 2>&1
```

Esto hace un dump diario a las 04:30, comprimido, retiene 14 días.

### Datos (`data/`)

Los datos son reproducibles: el refresh los regenera. Pero para no perder el
historial de predicciones (el track record) si el disco muere:

```bash
# Desde el PC con Windows, periódicamente (semanal basta):
scp mundialytics@<IP>:/srv/mundialytics/data/processed/logs/predictions_log.csv D:\mundialytics-backups\
scp mundialytics@<IP>:/srv/mundialytics/backups/*.sql.gz D:\mundialytics-backups\
```

---

## 8. El refresh diario

### Qué hace

El contenedor `refresh` es el equivalente de la tarea programada de Windows.
Duerme hasta `REFRESH_AT` (10:00 hora de Madrid), ejecuta
`scripts/update_season.py`, invalida la caché de la web, arranca el lineup
watcher y vuelve a dormir.

### Verificar que funciona (primera vez)

```bash
# Forzar un refresh inmediato para validar
docker compose exec refresh python scripts/update_season.py

# O, si quieres que lo haga el loop:
docker compose stop refresh
REFRESH_ON_START=1 docker compose up -d refresh
docker compose logs -f refresh
```

### Lo que el refresh NO hace en la nube

**El commit del track record.** `scripts/commit_track_record.py` commitea
`predictions_log.csv` y el drift report a git, pero el contenedor no tiene
checkout ni credenciales. Las opciones son:

1. **(Recomendada)** Seguir ejecutando `commit_track_record.py` en el PC contra
   una copia sincronizada con `scp`
2. Montar un deploy key de solo lectura y un cron en el VPS que haga el commit +
   push (más complejo, se pierde la regla de "no push automático")

---

## 9. Actualizaciones del código

Cuando hay cambios en el repo:

```bash
cd /srv/mundialytics
git pull

# Si cambiaron los Dockerfiles o las dependencias:
docker compose build api
docker buildx bake -f docker-compose.yml web --allow=network.host --load

# Aplicar
docker compose up -d
```

Si solo cambió el código Python (no dependencias ni Dockerfile):

```bash
git pull
docker compose build api
docker compose up -d api refresh
```

---

## 10. Monitorización (mínima pero real)

### Health checks integrados

Cada servicio tiene su propio healthcheck en docker-compose.yml. Docker los
ejecuta y reinicia el contenedor si falla.

### last_run.json

El refresh escribe `data/processed/logs/last_run.json` después de cada
ejecución. Un script o cron que lo lea puede alertar si `ok: false` o si
`finished_at` lleva más de 36 horas sin actualizarse.

### Ejemplo de alerta básica por email

```bash
# /srv/mundialytics/scripts/check_refresh.sh
#!/bin/sh
LAST=$(python3 -c "
import json, sys
from datetime import datetime, timedelta
d = json.load(open('/srv/mundialytics/data/processed/logs/last_run.json'))
t = datetime.fromisoformat(d['finished_at'])
if datetime.now() - t > timedelta(hours=36):
    print('STALE')
    sys.exit(1)
if not d['ok']:
    print('FAILED')
    sys.exit(1)
print('OK')
")
if [ $? -ne 0 ]; then
  echo "Mundialytics refresh: $LAST" | mail -s "Mundialytics alert" tu@email.com
fi
```

```cron
0 12 * * *  /srv/mundialytics/scripts/check_refresh.sh
```

---

## 11. Referencia: comparación con PropsAI

PropsAI usa una arquitectura completamente distinta que **no aplica** aquí:

| | PropsAI | Mundialytics |
|---|---|---|
| Compute | GitHub Actions (cron) | VPS con Docker Compose |
| Web | Vercel (2 ficheros estáticos) | Next.js SSR con prerender |
| API | No hay (todo estático) | FastAPI con ~200 MB de modelos en RAM |
| Datos | SQLite exportado a CSV | 2,6 GB de CSVs + caché de modelos |
| Auth | No hay | Better Auth + Postgres |

PropsAI puede vivir en servicios gratuitos porque es un pipeline batch que
produce dos ficheros. Mundialytics necesita un servidor con estado (modelos en
memoria, Postgres, SSR) y por eso el VPS es la decisión correcta.

---

## 12. Checklist de despliegue

- [ ] Alquilar VPS Hetzner CX32
- [ ] Asegurar el servidor (SSH, firewall, usuario no-root)
- [ ] Instalar Docker + Docker Compose
- [ ] Clonar el repo
- [ ] Subir `data/` (opción B: rsync sin data/raw)
- [ ] Crear `.env` con todos los valores
- [ ] Levantar postgres + api (`docker compose up -d --wait postgres api`)
- [ ] Construir web (`docker buildx bake ...`)
- [ ] Levantar todo (`docker compose up -d`)
- [ ] Verificar que la web carga y la API responde
- [ ] Comprar dominio y apuntar DNS
- [ ] Verificar TLS automático
- [ ] Instalar cron de backup de Postgres
- [ ] Ejecutar un refresh manual y verificar que completa
- [ ] Esperar al refresh automático del día siguiente y verificar logs
- [ ] Configurar alerta de refresh (opcional)
- [ ] Desactivar la tarea programada de Windows (una vez estable)

---

## 13. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Cold start de 5-20 min en 2 vCPU | Primera visita tras reinicio ve "warming up" | `start_period: 1200s` en el healthcheck; el refresh mantiene la caché caliente |
| ESPN/Sofascore bloquean la IP del VPS | El refresh falla en pasos opcionales | Pasos opcionales no rompen la fundación; el motor sirve con datos del día anterior |
| Disco lleno por `data/raw` creciente | Stack deja de funcionar | `data/raw` no se copia; el refresh no escribe ahí |
| Se olvida renovar el dominio | El sitio deja de resolver | Renovación automática en la mayoría de registradores |
| Pérdida del VPS | Se pierden las cuentas de usuario | Backup diario de Postgres + copia semanal al PC |
