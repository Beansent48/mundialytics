# Errores típicos al pasar del curso en R a Python

## 1. `glm(..., family = poisson())` vs Python

En R:

```r
glm(Goals ~ X1 + X2, family = poisson(), data = df)
```

En Python tienes dos opciones principales:

```python
from sklearn.linear_model import PoissonRegressor
```

O:

```python
import statsmodels.api as sm
```

Puntos importantes:

- En Poisson GLM, la predicción útil para nuestro bot es la **media esperada positiva**, es decir `lambda`.
- En statsmodels, `model.predict()` normalmente devuelve media en escala respuesta si el modelo está bien especificado, pero hay que evitar confundirla con el predictor lineal/log-lambda.
- En sklearn `PoissonRegressor` usa enlace log y `.predict()` devuelve valores positivos.

## 2. Variables categóricas

R maneja factores de forma muy cómoda. En Python hay que codificar categóricas explícitamente:

```python
OneHotEncoder(handle_unknown="ignore")
```

En este proyecto se hace con `ColumnTransformer` dentro del `GoalLambdaModel`.

## 3. Índices 1-based vs 0-based

R empieza en 1. Python empieza en 0. En el Lab 3 se recorren filas de dos en dos para Team1/Team2. En Python no conviene asumir que las filas siempre vienen emparejadas y ordenadas. Por eso este proyecto usa:

```python
to_long_team_rows(matches)
```

Así genera explícitamente las dos observaciones por partido.

## 4. Skellam: no truncar mal las colas

En el lab se ve una aproximación sumando `dskellam(1:10)` y `dskellam(-10:-1)`. En Python podemos usar directamente CDF:

```python
from scipy.stats import skellam

p_home = 1 - skellam.cdf(0, lambda_home, lambda_away)
p_draw = skellam.pmf(0, lambda_home, lambda_away)
p_away = skellam.cdf(-1, lambda_home, lambda_away)
```

Esto evita perder masa de probabilidad por truncar demasiado pronto.

## 5. Random Forest no garantiza lambdas positivas

Si un Random Forest predice goles esperados, puede dar valores bajos o incluso raros en datos ruidosos. Por eso se debe aplicar:

```python
lambda_hat = clip(lambda_hat, floor=0.05, cap=6.0)
```

El PoissonRegressor, en cambio, ya mantiene predicciones positivas.

## 6. Overfitting

Los árboles y Random Forests capturan no linealidades, pero también pueden aprender ruido si no se validan bien. Para torneos, la validación correcta no es un random split cualquiera, sino:

- Leave-one-tournament-out.
- Rolling window temporal.
- Backtesting por fecha.

## 7. Fuga temporal de datos

Nunca uses estadísticas del propio partido para predecir ese partido. Las rolling features deben ir con `shift(1)`.

Este proyecto lo hace en:

```python
add_pre_match_rolling_features()
```

## 8. Cuotas y probabilidad implícita

Con cuota decimal:

```python
p_implicita = 1 / cuota
```

Pero en mercados con varias selecciones hay margen/overround. Para 1X2 conviene normalizar las probabilidades implícitas por mercado. En player props individuales normalmente falta la cuota del lado contrario, así que el cálculo es más aproximado.

## 9. Sustituto+

No asumir reglas universales. Este proyecto usa una hipótesis práctica para threshold markets:

```text
Gana si jugador original O sustituto cumplen la línea.
```

Antes de usarlo fuera de paper mode, revisar reglas concretas de la casa y del mercado.
