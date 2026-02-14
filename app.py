import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="Analyse dataset (Histogramme, Âge, Normalité, Déséquilibre)",
    layout="wide",
)

# =========================
# Paramètres FIXES (pas d'auto-détection)
# =========================
CSV_PATH_DEFAULT = (
    "data_corriger_new.csv"  # mets ton CSV dans le même dossier que app.py
)

ID_COLS = ["Nom", "Prenom"]
AGE_COL = "Age"
AGE_CAT_COL = "Tranche_age"

LSI_COLS = [
    "Wall_test_LSI",
    "Side_hop_LSI",
    "Decolle_talon_LSI",
    "Assis_debout_LSI",
    "Y_balance_LSI",
]

TARGET_MAP = {
    "Wall_test_LSI": "Target_Wall_test_max",
    "Side_hop_LSI": "Target_Side_hop_min",
    "Decolle_talon_LSI": "Target_Decolle_talon_min",
    "Assis_debout_LSI": "Target_Assis_debout_min",
    "Y_balance_LSI": "Target_Y_balance_min",
}

REQUIRED_COLS = ID_COLS + [AGE_COL, AGE_CAT_COL] + LSI_COLS + list(TARGET_MAP.values())


# =========================
# Helpers
# =========================
def load_df(uploaded_file):
    if uploaded_file is not None:
        # ton fichier est en ;, on respecte ça
        return pd.read_csv(uploaded_file, sep=";")
    # fallback local
    return pd.read_csv(CSV_PATH_DEFAULT, sep=";")


def check_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def normality_block(x: pd.Series, title_prefix: str = ""):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3:
        st.warning("Pas assez de données numériques pour tester la normalité.")
        return

    colA, colB = st.columns([1.2, 1])

    with colA:
        fig = plt.figure()
        mu = float(x.mean())
        sigma = float(x.std(ddof=1)) if float(x.std(ddof=1)) > 0 else 1.0
        plt.hist(x, density=True)
        xs = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(xs, stats.norm.pdf(xs, mu, sigma))
        plt.title(f"{title_prefix}Histogramme + loi normale ajustée")
        st.pyplot(fig, clear_figure=True)

    with colB:
        fig = plt.figure()
        stats.probplot(x, dist="norm", plot=plt)
        plt.title(f"{title_prefix}Q-Q plot (normalité)")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Tests de normalité (indicatifs)")
    rows = []

    if len(x) <= 5000:
        sh = stats.shapiro(x)
        rows.append(
            {"Test": "Shapiro-Wilk", "Statistique": sh.statistic, "p-value": sh.pvalue}
        )

    if len(x) >= 8:
        dag = stats.normaltest(x)
        rows.append(
            {
                "Test": "D’Agostino K²",
                "Statistique": dag.statistic,
                "p-value": dag.pvalue,
            }
        )

    if not rows:
        st.info("Pas assez de données pour exécuter les tests.")
    else:
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"Statistique": "{:.4f}", "p-value": "{:.4g}"}
            ),
            use_container_width=True,
        )
        st.caption("Règle pratique: si p-value < 0.05, la normalité est peu probable.")


def bland_altman_vs_target(df: pd.DataFrame, lsi_col: str, target_col: str):
    a = pd.to_numeric(df[lsi_col], errors="coerce")
    b = pd.to_numeric(df[target_col], errors="coerce")
    tmp = df[ID_COLS + [AGE_COL, AGE_CAT_COL]].copy()
    tmp["LSI"] = a
    tmp["Target"] = b
    tmp = tmp.dropna(subset=["LSI", "Target"])

    if tmp.empty:
        st.warning("Pas assez de valeurs numériques pour Bland-Altman.")
        return

    diff = tmp["LSI"] - tmp["Target"]
    mean = (tmp["LSI"] + tmp["Target"]) / 2

    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if float(diff.std(ddof=1)) > 0 else 0.0
    loa_hi = md + 1.96 * sd
    loa_lo = md - 1.96 * sd

    fig = plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, linestyle="--")
    plt.axhline(loa_hi, linestyle="--")
    plt.axhline(loa_lo, linestyle="--")
    plt.title(f"Bland-Altman : {lsi_col} vs {target_col} (diff = LSI - Target)")
    plt.xlabel("Moyenne (LSI + Target)/2")
    plt.ylabel("Différence (LSI - Target)")
    st.pyplot(fig, clear_figure=True)

    st.write(
        f"**Biais moyen (diff moyenne)**: {md:.4g}  \n"
        f"**Écart-type des diff**: {sd:.4g}  \n"
        f"**Limites d’accord (±1.96 SD)**: [{loa_lo:.4g}, {loa_hi:.4g}]"
    )

    # Définition "gros déséquilibre" basée sur min/max du Target
    is_min = target_col.lower().endswith("_min")
    is_max = target_col.lower().endswith("_max")

    if is_min:
        fail = tmp["LSI"] < tmp["Target"]
        rule = "Fail si LSI < Target (objectif minimum)"
    elif is_max:
        fail = tmp["LSI"] > tmp["Target"]
        rule = "Fail si LSI > Target (objectif maximum)"
    else:
        # fallback: on prend diff négatif
        fail = diff < 0
        rule = "Fail si LSI - Target < 0"

    st.caption(f"Règle utilisée : {rule}")

    n_fail = int(fail.sum())
    st.write(f"**Nombre d’agents en déséquilibre (fail)**: {n_fail} / {len(tmp)}")

    if n_fail > 0:
        out = tmp.loc[fail].copy()
        out["Diff(LSI-Target)"] = (out["LSI"] - out["Target"]).values
        out["Moyenne"] = ((out["LSI"] + out["Target"]) / 2).values
        st.dataframe(out.sort_values("Diff(LSI-Target)"), use_container_width=True)


# =========================
# UI - Data input
# =========================
st.sidebar.header("Données")
uploaded = st.sidebar.file_uploader("Uploader ton CSV (séparateur ;)", type=["csv"])
df = load_df(uploaded)

missing = check_required_columns(df)
if missing:
    st.error("Colonnes manquantes dans le dataset : " + ", ".join(missing))
    st.stop()

st.subheader("Aperçu du dataset")
st.dataframe(df.head(20), use_container_width=True)

# =========================
# Sélections (fixes, basées sur ce dataset)
# =========================
st.sidebar.header("Choix analyse")
mode = st.sidebar.radio(
    "Variable à analyser", ["LSI (valeur brute)", "Déséquilibre = |100 - LSI|"], index=0
)

lsi_choice = st.sidebar.selectbox("Quel test LSI ?", LSI_COLS, index=0)

# Série analysée (histogramme / scatter / normalité)
x_raw = pd.to_numeric(df[lsi_choice], errors="coerce")
if mode == "LSI (valeur brute)":
    series_to_analyze = x_raw
    y_label = lsi_choice
else:
    series_to_analyze = (100 - x_raw).abs()
    y_label = f"|100 - {lsi_choice}|"

# =========================
# 1) Histogramme
# =========================
st.header("1) Histogramme — répartition des valeurs")
vals = series_to_analyze.dropna()
fig = plt.figure()
plt.hist(vals)
plt.title(f"Histogramme: {y_label}")
plt.xlabel(y_label)
plt.ylabel("Effectif")
st.pyplot(fig, clear_figure=True)

# =========================
# 2) Nuage de points vs âge (catégories)
# =========================
st.header("2) Nuage de points — en fonction de l'âge (avec Tranche_age)")
plot_df = df[ID_COLS + [AGE_COL, AGE_CAT_COL]].copy()
plot_df["val"] = series_to_analyze
plot_df[AGE_COL] = pd.to_numeric(plot_df[AGE_COL], errors="coerce")
plot_df = plot_df.dropna(subset=[AGE_COL, "val", AGE_CAT_COL])

if plot_df.empty:
    st.warning("Pas assez de données pour le nuage de points.")
else:
    fig = plt.figure()
    for cat, g in plot_df.groupby(AGE_CAT_COL):
        plt.scatter(g[AGE_COL], g["val"], label=str(cat))
    plt.title(f"{y_label} vs Âge (catégories = {AGE_CAT_COL})")
    plt.xlabel("Âge")
    plt.ylabel(y_label)
    plt.legend(title="Tranche d'âge", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)

# =========================
# 3) Normalité
# =========================
st.header("3) Normalité — loi normale ou pas ?")
normality_block(series_to_analyze, title_prefix=f"{y_label} — ")

# =========================
# 4) Bland-Altman : LSI vs Target
# =========================
st.header("4) Diagramme de Bland-Altman — LSI vs Target (seuil du test)")
target_col = TARGET_MAP[lsi_choice]
bland_altman_vs_target(df, lsi_choice, target_col)

# =========================
# Bonus : tableau résumé pass/fail sur tous les tests
# =========================
st.header("Résumé — Pass/Fail par test (selon Target min/max)")
summary = df[ID_COLS + [AGE_COL, AGE_CAT_COL]].copy()

for lsi_col, tcol in TARGET_MAP.items():
    lsi = pd.to_numeric(df[lsi_col], errors="coerce")
    tgt = pd.to_numeric(df[tcol], errors="coerce")
    is_min = tcol.lower().endswith("_min")
    is_max = tcol.lower().endswith("_max")

    if is_min:
        fail = lsi < tgt
    elif is_max:
        fail = lsi > tgt
    else:
        fail = (lsi - tgt) < 0

    summary[f"{lsi_col}_FAIL"] = fail

st.dataframe(summary, use_container_width=True)
