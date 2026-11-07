"""
задание 4.5:
- Модель загружается через st.cache_resource. Этот декоратор хранит один объект на весь сервер
  и отдаёт его всем вызовам и всем пользователям без копирования. Для модели это то, что нужно:
  она большая и тяжело собирается из байтов, а predict_proba её не меняет
- Таблицы (test.csv, вероятности на test, точки ROC/PR-кривых) кэшируются через st.cache_data.
  Он хранит результат в сериализованном виде и каждому вызову отдаёт копию. Так один пользователь
  не испортит таблицу остальным, если что-то в ней поменяет.
  Модель через st.cache_data грузить не стоит: на каждый вызов она бы заново собиралась из байтов
  (по сути тот же joblib.load), плюс Streamlit пытался бы её хешировать.
- Если вызвать joblib.load без декоратора, модель будет читаться с диска при каждом перезапуске
  скрипта, т.е. на каждое движение слайдера или ввод числа. Приложение будет работать медленнее,
  и у каждой сессии будет своя копия модели в памяти.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

st.set_page_config(page_title="Подборка вин", page_icon="🍷")

# пути строим от папки, где лежит app.py, а не от текущей директории
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

# подпись, шаг и формат поля для каждого числового признака
FEATURES = {
    "fixed_acidity": ("Фиксированная кислотность, г/л", 0.1, "%.1f"),
    "volatile_acidity": ("Летучая кислотность, г/л", 0.01, "%.2f"),
    "citric_acid": ("Лимонная кислота, г/л", 0.01, "%.2f"),
    "residual_sugar": ("Остаточный сахар, г/л", 0.1, "%.1f"),
    "chlorides": ("Хлориды, г/л", 0.001, "%.3f"),
    "free_sulfur_dioxide": ("Свободный SO₂, мг/л", 1.0, "%.1f"),
    "total_sulfur_dioxide": ("Общий SO₂, мг/л", 1.0, "%.1f"),
    "density": ("Плотность, г/см³", 0.0001, "%.4f"),
    "pH": ("pH", 0.01, "%.2f"),
    "sulphates": ("Сульфаты, г/л", 0.01, "%.2f"),
    "alcohol": ("Алкоголь, %", 0.1, "%.1f"),
}


@st.cache_resource  # один объект модели на весь сервер
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text())


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    # _model с подчёркиванием: Streamlit не хеширует модель, хеширует только X
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    # точки кривых от порога не зависят, считаем один раз
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y, proba)
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": metrics.roc_auc_score(y, proba),
        "precision": pr_precision, "recall": pr_recall, "pr_auc": metrics.average_precision_score(y, proba),
    }


def read_batch(uploaded, feature_cols):
    """Читает загруженный CSV и проверяет его. Возвращает (таблица, признаки, текст ошибки или None)."""
    try:
        batch = pd.read_csv(uploaded)
    except Exception as e:
        return None, None, f"Не удалось прочитать CSV: {e}"
    if batch.empty:
        return None, None, "Файл пустой: в нём нет ни одной строки."
    missing = [col for col in feature_cols if col not in batch.columns]
    if missing:
        return None, None, f"В файле не хватает столбцов: {', '.join(missing)}"
    features_df = batch[feature_cols].apply(pd.to_numeric, errors="coerce")
    bad_cols = features_df.columns[features_df.isna().any()].tolist()
    if bad_cols:
        return None, None, f"Пустые или нечисловые значения в столбцах: {', '.join(bad_cols)}"
    return batch, features_df, None


missing_files = [path.name for path in [MODEL_PATH, CONFIG_PATH, TEST_PATH] if not path.exists()]
if missing_files:
    st.error(f"Нет файлов: {', '.join(missing_files)}. Сначала выполните ноутбук HW3_MMDA.ipynb.")
    st.stop()

model = load_model()
config = load_config()
test_df = load_test()
feature_cols = list(model.feature_names_in_)  # столбцы, на которых училась модель
X_test = test_df[feature_cols]
y_test = test_df["target"].to_numpy()
proba_test = test_proba(model, X_test)

st.title("🍷 Подборка хороших вин")
st.caption("Логистическая регрессия по 12 лабораторным признакам. «Хорошее» вино - оценка экспертов не ниже 7.")

threshold = st.sidebar.slider(
    "Порог «хорошее вино»", 0.0, 1.0, float(config["threshold"]), step=0.001, format="%.3f", key="threshold")
st.sidebar.caption(
    f"Порог по умолчанию {config['threshold']:.3f} подобран на валидации (максимум F1). "
    "Вино попадает в подборку, если вероятность не меньше порога.")

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])

# --- 4.1. Оценка одного вина ---
with tab_one:
    st.subheader("Характеристики вина")
    st.caption("Диапазоны - минимум и максимум по тестовой выборке, по умолчанию стоит медиана.")

    wine_type = st.radio("Тип вина", ["белое", "красное"], horizontal=True)
    wine = {}
    columns = st.columns(2)
    for i, (feature, (label, step, fmt)) in enumerate(FEATURES.items()):
        values = test_df[feature]
        with columns[i % 2]:
            wine[feature] = st.number_input(
                label, min_value=float(values.min()), max_value=float(values.max()),
                value=float(values.median()), step=step, format=fmt)
    wine["is_red"] = 1 if wine_type == "красное" else 0

    wine_df = pd.DataFrame([wine])[feature_cols]
    proba = model.predict_proba(wine_df)[0, 1]

    st.divider()
    left, right = st.columns(2)
    left.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
    right.metric("Порог", f"{threshold:.3f}")
    st.progress(float(proba))
    if proba >= threshold:
        st.success("Хорошее вино - берём в подборку.")
    else:
        st.warning("Скорее посредственное - в подборку не берём.")

# --- 4.2. Модельные метрики ---
with tab_metrics:
    st.subheader("Качество на тестовой выборке")
    pred_test = (proba_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, pred_test, zero_division=0)
    recall = metrics.recall_score(y_test, pred_test)
    f1 = metrics.f1_score(y_test, pred_test, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")
    c4.metric("Вин в подборке", f"{tp + fp} из {len(y_test)}")

    st.info(
        f"Из {tp + fp} вин в подборке действительно хороших {tp} ({precision:.0%}). "
        f"Из {tp + fn} хороших вин в test в подборку попали {tp} ({recall:.0%}).")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["не хорошее", "хорошее"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Решение модели")
    ax.set_ylabel("Оценка экспертов")
    ax.set_title("Матрица ошибок")
    st.pyplot(fig)
    plt.close(fig)

    curves = test_curves(y_test, proba_test)
    fpr_point = fp / (fp + tn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(curves["fpr"], curves["tpr"], label=f"ROC-AUC = {curves['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Случайная модель")
    axes[0].scatter(fpr_point, recall, s=100, color="red", edgecolor="black", zorder=3, label=f"Порог {threshold:.3f}")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR (Recall)")
    axes[0].set_title("ROC-кривая")
    axes[0].legend(loc="lower right")

    axes[1].plot(curves["recall"], curves["precision"], label=f"PR-AUC = {curves['pr_auc']:.3f}")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", linewidth=1, label="Случайная модель")
    if tp + fp > 0:  # если подборка пустая, precision не определён и точки нет
        axes[1].scatter(recall, precision, s=100, color="red", edgecolor="black", zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR-кривая")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "**Для сомелье: как порог меняет подборку**\n\n"
        "- **Порог выше** - модель берёт только вина, в которых очень уверена. Подборка станет короче, "
        "посредственных вин в ней будет меньше (precision растёт), но часть хороших вин мы пропустим (recall падает).\n"
        "- **Порог ниже** - подборка станет шире и в неё попадёт больше хороших вин (recall растёт), "
        "но вместе с ними и больше посредственных (precision падает).\n\n"
        "Красная точка на кривых показывает, где мы находимся при текущем пороге.")

# --- 4.3. Пакетное предсказание ---
with tab_batch:
    st.subheader("Прогноз по CSV-файлу")
    st.caption(f"Нужны столбцы: {', '.join(feature_cols)}. Порог берётся из слайдера слева ({threshold:.3f}).")
    st.download_button(
        "Скачать пример файла (20 вин из test.csv без target)",
        data=test_df[feature_cols].head(20).to_csv(index=False).encode("utf-8"),
        file_name="wines_example.csv", mime="text/csv", on_click="ignore")

    uploaded = st.file_uploader("CSV с признаками вин", type="csv")
    if uploaded is None:
        st.info("Загрузите CSV-файл с признаками вин.")
    else:
        batch, features_df, error = read_batch(uploaded, feature_cols)
        if error:
            st.error(error)
        else:
            result = batch.copy()
            result["proba_good"] = model.predict_proba(features_df)[:, 1]
            result["in_selection"] = (result["proba_good"] >= threshold).astype(int)
            st.metric("Вин в подборке", f"{result['in_selection'].sum()} из {len(result)}")
            st.dataframe(
                result.sort_values("proba_good", ascending=False), hide_index=True,
                column_config={"proba_good": st.column_config.ProgressColumn(
                    "Вероятность", format="%.3f", min_value=0.0, max_value=1.0)})
            st.download_button(
                "Скачать прогнозы", data=result.to_csv(index=False).encode("utf-8"),
                file_name="wine_predictions.csv", mime="text/csv", on_click="ignore")

# --- 4.4. Что выучила модель ---
with tab_coef:
    st.subheader("Коэффициенты логистической регрессии")
    st.caption("Признаки стандартизированы, поэтому коэффициенты можно сравнивать: это изменение "
               "логарифма шансов «хорошего» вина при росте признака на одно стандартное отклонение.")
    coefs = pd.Series(model[-1].coef_[0], index=feature_cols).sort_values()
    labels = {feature: label for feature, (label, _, _) in FEATURES.items()} | {"is_red": "Красное вино"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([labels[f] for f in coefs.index], coefs.values,
            color=["tab:green" if c > 0 else "tab:red" for c in coefs.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент")
    ax.set_title("Зелёные голосуют за хорошее вино, красные - против")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_plus = coefs[coefs > 0].sort_values(ascending=False).head(3)
    top_minus = coefs[coefs < 0].head(3)
    st.markdown(
        "**Сильнее всего за хорошее вино:** "
        + ", ".join(f"{labels[f]} ({c:+.2f})" for f, c in top_plus.items())
        + "\n\n**Сильнее всего против:** "
        + ", ".join(f"{labels[f]} ({c:+.2f})" for f, c in top_minus.items()))
