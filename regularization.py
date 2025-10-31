import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {"alpha": np.logspace(-3, 3, 20)}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {"alpha": np.logspace(-3, 1, 20)}
    lasso = Lasso(max_iter=5000)
    grid = GridSearchCV(lasso, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model = LogisticRegression(penalty=None, max_iter=5000)
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {"C": np.logspace(-3, 3, 20)}
    model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    param_grid = {"C": np.logspace(-3, 3, 20)}
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_


# ==================  RESULTS & CONCLUSIONS ==================
"""Linear Regression vs Ridge vs Lasso
LinearRegression -> найпростіша модель, може перенавчатися, без регуляризації
Звичайна лінійна регресія просто будує пряму/гіперплощину без контролю складності, тому може "підлаштуватись" під шум у даних.

Ridge -> L2-регуляризація, зменшує коефіцієнти, стабільна при мультиколінеарності
Ridge згладжує модель, робить ваги меншими, краще працює, коли ознаки сильно корелюють між собою.

Lasso -> L1-регуляризація, може обнулювати коефіцієнти -> виконує відбір ознак
Lasso не просто робить ваги меншими — воно може "вимикати" зайві ознаки, тому корисне для вибору важливих предикторів.

Очікувана поведінка на наборі даних diabetes:
Lasso може відкинути частину ознак -> компактніша модель
Ridge зазвичай забезпечує найкращу узагальнюючу здатність (меншу помилку на тесті)
Linear Regression може давати більшу помилку через відсутність регуляризації
Lasso робить модель "легшою", Ridge — найстабільніший варіант, Linear Regression програє на реальних даних із шумом.

Logistic Regression models on breast cancer dataset
Logistic без регуляризації -> базова модель
Просто логістична регресія без захисту від перенавчання.

Logistic L2 -> найстабільніша, захищає від перенавчання
Найчастіше найкращий варіант для класифікації.

Logistic L1 -> може видаляти слабкі ознаки -> найкраще для розріджених моделей
Корисно, коли багато ознак і треба вибрати найважливіші.

Logistic L2 ≥ Logistic без регуляризації ≥ Logistic L1
(але якщо багато зайвих ознак — L1 може виграти)
Пояснення:
L2 — найчастіше найкращий.
L1 перемагає там, де багато непотрібних ознак.

Overall:
Ridge та логістична регресія з L2 → найкраще узагальнення
Lasso -> корисна для відбору ознак
Лінійні моделі без регуляризації → ризик перенавчання
Пояснення:
Ridge — стабільний чемпіон,
Lasso — обирає найкращі ознаки,
без регуляризації = небезпека перепідгонки."""
