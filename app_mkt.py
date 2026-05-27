"""
Segmentación de Clientes: E-commerce Global
Trabajo 2 Marketing 2026-1 — Universidad de Concepción
GMM (RFM) + K-Prototypes (Demográfico)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import warnings
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kmodes.kprototypes import KPrototypes
import plotly.express as px

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación E-commerce | UdeC",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.hero-box {
    background: linear-gradient(135deg, #0a2342 0%, #1a3a5c 60%, #0e4f7a 100%);
    border-radius: 16px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 32px;
    border-left: 6px solid #f5a623;
}
.hero-box h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.5px; }
.hero-box p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }
.hero-badge {
    display: inline-block;
    background: #f5a623;
    color: #1a6fb5;
    font-weight: 700;
    font-size: 0.78rem;
    border-radius: 20px;
    padding: 4px 14px;
    margin-bottom: 18px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.section-header {
    border-left: 5px solid #f5a623;
    padding-left: 16px;
    margin: 28px 0 20px 0;
}
.section-header h2 { margin: 0; font-weight: 700; font-size: 1.6rem; color: #1a6fb5; }
.section-header p  { margin: 4px 0 0 0; color: e0e0e0; font-size: 0.95rem; }

.kpi-card {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 14px;
    text-align: center;
    border-top: 4px solid #0a2342;;
}
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; color: #1a6fb5; }
.kpi-card .kpi-label { font-size: 0.85rem; color: #666; margin-top: 4px; }

.segment-card {
    border-radius: 12px;
    padding: 18px 16px;
    margin-bottom: 12px;
    border-left: 5px solid;
    background: #fdfdfd;
}
.info-box {
    background: #eef4fb;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #1a6fb5;
    margin: 16px 0;
    font-size: 0.93rem;
    color: #2c3e50;
}
.warn-box {
    background: #fff8ec;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #f5a623;
    margin: 16px 0;
    font-size: 0.93rem;
    color: #7a4a00;
}

.sidebar-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #1a6fb5;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🛒 Segmentación E-commerce</p>', unsafe_allow_html=True)
    st.caption("Universidad de Concepción · Marketing 2026-1")
    st.divider()
    seccion = st.radio(
        "Sección",
        options=[
            "🏠  Contexto del Mercado",
            "📋  Descripción de Datos",
            "🧹  Limpieza de Datos",
            "📈  Modelo RFM — GMM",
            "👥  Modelo Demográfico — K-Prototypes",
            "🔗  Análisis de Fusión",
            "🎯  Conclusiones y Posicionamiento",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Trabajo 2 · Mercado 1: E-commerce global")


# ─────────────────────────────────────────────
# CARGA DE DATOS (CACHEADO)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def cargar_datos():
    df_customers = pd.read_csv("customers.csv")
    df_orders    = pd.read_csv("orders.csv")
    return df_customers, df_orders


# ─────────────────────────────────────────────
# PIPELINE COMPLETO (CACHEADO) — GMM + KPROTO + FUSIÓN
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Ejecutando modelos de segmentación…")
def ejecutar_pipeline(_df_customers, _df_orders):
    df_customers = _df_customers.copy()
    df_orders    = _df_orders.copy()

    Frecuencia = "total_orders"
    Monetario  = "avg_order_value_usd"
    Recencia   = "days_since_last_purchase"

    # ── Limpieza RFM ──────────────────────────
    df_RFM = df_customers[["customer_id", Recencia, Frecuencia, Monetario]].copy()
    df_RFM = df_RFM[df_RFM["customer_id"].isin(df_orders["customer_id"])]
    df_RFM = df_RFM[(df_RFM[Frecuencia] > 0) & (df_RFM[Monetario] > 0) & (df_RFM[Recencia] >= 0)]

    # Winsorización 1%–98%
    for col in [Frecuencia, Monetario, Recencia]:
        lo, hi = df_RFM[col].quantile(0.01), df_RFM[col].quantile(0.98)
        df_RFM[col] = df_RFM[col].clip(lo, hi)

    # ── GMM — selección de k ──────────────────
    scaler_gmm = MinMaxScaler()
    X_gmm = scaler_gmm.fit_transform(df_RFM[[Frecuencia, Monetario, Recencia]])

    metricas_gmm = {"k": [], "BIC": [], "AIC": [], "Silueta": [], "DBI": [], "CHI": [], "Entropia": [], "LogLik": []}
    for k in range(2, 11):
        gm = GaussianMixture(n_components=k, n_init=10, random_state=42)
        gm.fit(X_gmm)
        lbl = gm.predict(X_gmm)
        prb = gm.predict_proba(X_gmm)
        e_bruta = np.mean(entropy(prb, axis=1))
        e_esc   = 1 - (e_bruta / np.log(k))
        metricas_gmm["k"].append(k)
        metricas_gmm["BIC"].append(gm.bic(X_gmm))
        metricas_gmm["AIC"].append(gm.aic(X_gmm))
        metricas_gmm["Silueta"].append(silhouette_score(X_gmm, lbl))
        metricas_gmm["DBI"].append(davies_bouldin_score(X_gmm, lbl))
        metricas_gmm["CHI"].append(calinski_harabasz_score(X_gmm, lbl))
        metricas_gmm["Entropia"].append(e_esc)
        metricas_gmm["LogLik"].append(gm.score(X_gmm))

    # ── GMM definitivo k=4 ────────────────────
    df_GMM = df_RFM[["customer_id", Frecuencia, Monetario, Recencia]].copy()
    GMM = GaussianMixture(n_components=4, n_init=10, random_state=42)
    GMM.fit(X_gmm)
    df_GMM["Cluster"] = GMM.predict(X_gmm)

    # ── K-Prototypes — preparación ────────────
    lista_dem = ["age", "gender", "membership_tier"]
    df_dem = df_RFM[["customer_id"]].merge(
        df_customers[["customer_id"] + lista_dem], on="customer_id", how="left"
    )
    dic_gender = {"Female": 0, "Male": 1, "Other": 2}
    dic_tier   = {"Free": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    df_dem["gender"]          = df_dem["gender"].map(dic_gender)
    df_dem["membership_tier"] = df_dem["membership_tier"].map(dic_tier)

    # ── KProto — selección de k ───────────────
    scaler_dem = MinMaxScaler()
    X_dem = df_dem[lista_dem].copy()
    X_dem["age"] = scaler_dem.fit_transform(df_dem[["age"]])
    X_kp = X_dem.values
    cat_idx = [1, 2]

    metricas_kp = {"k": [], "Costo": [], "Silueta": [], "DBI": [], "CHI": []}
    for k in range(2, 11):
        kp = KPrototypes(n_clusters=k, init="Cao", random_state=42, n_init=5, verbose=0)
        lbl = kp.fit_predict(X_kp, categorical=cat_idx)
        metricas_kp["k"].append(k)
        metricas_kp["Costo"].append(kp.cost_)
        metricas_kp["Silueta"].append(silhouette_score(X_kp, lbl))
        metricas_kp["DBI"].append(davies_bouldin_score(X_kp, lbl))
        metricas_kp["CHI"].append(calinski_harabasz_score(X_kp, lbl))

    # ── KProto definitivo k=3 ─────────────────
    df_dem_final = df_dem[["customer_id"] + lista_dem].copy()
    X_dem_final = df_dem_final[lista_dem].copy()
    X_dem_final["age"] = scaler_dem.transform(df_dem_final[["age"]])
    kp_final = KPrototypes(n_clusters=3, init="Cao", random_state=42, n_init=10, verbose=0)
    df_dem_final["Cluster"] = kp_final.fit_predict(X_dem_final.values, categorical=cat_idx)

    # ── Fusión ────────────────────────────────
    # Nombres dinámicos RFM
    perf_rfm = df_GMM.groupby("Cluster").agg(
        Frecuencia=(Frecuencia, "mean"),
        Monetario =(Monetario,  "mean"),
        Recencia  =(Recencia,   "mean"),
    )
    raw_premium   = perf_rfm["Monetario"].idxmax()
    raw_historico = perf_rfm["Frecuencia"].idxmax()
    restantes = [c for c in perf_rfm.index if c not in [raw_premium, raw_historico]]
    raw_promedio  = perf_rfm.loc[restantes, "Recencia"].idxmin()
    raw_ocasional = [c for c in restantes if c != raw_promedio][0]
    nombres_rfm = {
        raw_ocasional: "Cliente Ocasional",
        raw_promedio:  "Cliente Promedio",
        raw_premium:   "Clientes Premium",
        raw_historico: "Clientes Históricos/Transaccionales",
    }

    # Nombres dinámicos demográficos
    perf_dem   = df_dem_final.groupby("Cluster")["age"].mean()
    orden_edad = perf_dem.sort_values().index
    nombres_dem = {
        orden_edad[0]: "Hombres Más Jóvenes",
        orden_edad[1]: "Mujeres Poder Adq. Medio-Bajo",
        orden_edad[2]: "Hombres Mayor Edad Poder Adq. Medio",
    }

    # Merge
    df_rfm_f = df_GMM[["customer_id", "Cluster"]].rename(columns={"Cluster": "Cluster_RFM"}).set_index("customer_id")
    df_dem_f = df_dem_final[["customer_id", "Cluster"]].rename(columns={"Cluster": "Cluster_DEM"}).set_index("customer_id")
    df_fusion = df_rfm_f.join(df_dem_f, how="inner")

    # Órdenes filtradas (solo con descuento > 0 para análisis de receptividad)
    df_ord = df_orders.copy()
    df_ord["order_date"] = pd.to_datetime(df_ord["order_date"])
    df_orders_discount = df_ord[df_ord["discount_pct"] > 0]

    df_ord_agg = df_ord.groupby("customer_id").agg(
        discount_pct_prom        =("discount_pct",                 "mean"),
        session_duration_prom    =("session_duration_minutes",     "mean"),
        pages_viewed_prom        =("pages_viewed_before_purchase",  "mean"),
        is_repeat_pct            =("is_repeat_customer",            "mean"),
        delivery_days_prom       =("delivery_days",                 "mean"),
        rating_prom              =("customer_rating",               "mean"),
        categoria_favorita       =("category",   lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        metodo_pago_favorito     =("payment_method", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        n_ordenes                =("order_id",                      "count"),
    ).reset_index().set_index("customer_id")

    df_cust_extra = df_customers.set_index("customer_id")[[
        "age", "gender", "membership_tier", "acquisition_channel",
        "returns_made", "avg_review_score", "wishlist_items", "churned",
    ]]
    df_fusion = df_fusion.join(df_cust_extra, how="left")
    df_fusion = df_fusion.join(df_ord_agg,    how="left")

    df_fusion["Nombre_RFM"] = df_fusion["Cluster_RFM"].map(nombres_rfm)
    df_fusion["Nombre_DEM"] = df_fusion["Cluster_DEM"].map(nombres_dem)
    df_fusion["Micro_Segmento"] = df_fusion["Nombre_RFM"] + " | " + df_fusion["Nombre_DEM"]

    # Revertir mapeos para categorías
    map_gender_rev = {0: "Female", 1: "Male", 2: "Other"}
    map_tier_rev   = {0: "Free", 1: "Silver", 2: "Gold", 3: "Platinum"}
    df_fusion_cat  = df_fusion.copy()
    if pd.api.types.is_numeric_dtype(df_fusion_cat["gender"]):
        df_fusion_cat["gender"]          = df_fusion_cat["gender"].map(map_gender_rev)
    if pd.api.types.is_numeric_dtype(df_fusion_cat["membership_tier"]):
        df_fusion_cat["membership_tier"] = df_fusion_cat["membership_tier"].map(map_tier_rev)

    # Órdenes con micro-segmento
    df_ord_seg = df_ord.merge(
        df_fusion[["Micro_Segmento"]].reset_index(), left_on="customer_id", right_on="customer_id", how="inner"
    )

    return {
        "df_RFM":        df_RFM,
        "df_GMM":        df_GMM,
        "metricas_gmm":  pd.DataFrame(metricas_gmm),
        "df_dem_final":  df_dem_final,
        "metricas_kp":   pd.DataFrame(metricas_kp),
        "df_fusion":     df_fusion,
        "df_fusion_cat": df_fusion_cat,
        "df_ord_seg":    df_ord_seg,
        "nombres_rfm":   nombres_rfm,
        "nombres_dem":   nombres_dem,
        "Frecuencia":    Frecuencia,
        "Monetario":     Monetario,
        "Recencia":      Recencia,
        "lista_dem":     ["age", "gender", "membership_tier"],
        "dic_gender":    dic_gender,
        "dic_tier":      dic_tier,
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(
        f"""<div class="section-header">
            <h2>{title}</h2>
            {"<p>" + subtitle + "</p>" if subtitle else ""}
        </div>""",
        unsafe_allow_html=True,
    )

CLUSTER_COLORS_RFM = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLUSTER_COLORS_DEM = ["#9b59b6", "#1abc9c", "#e67e22"]


def heatmap_anotado(data, ax, fmt=".2f", cmap="RdYlGn", title="", xlabel="", ylabel="", cbar_label=""):
    sns.heatmap(data, annot=False, cmap=cmap, linewidths=0.5, ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    vmin, vmax = data.values.min(), data.values.max()
    umbral = (vmin + vmax) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iloc[i, j]
            color = "white" if v > umbral else "black"
            ax.text(j + 0.5, i + 0.5, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color, clip_on=False)


# ══════════════════════════════════════════════
# CARGA Y EJECUCIÓN
# ══════════════════════════════════════════════
df_customers, df_orders = cargar_datos()
res = ejecutar_pipeline(df_customers, df_orders)

df_RFM       = res["df_RFM"]
df_GMM       = res["df_GMM"]
df_dem_final = res["df_dem_final"]
df_fusion    = res["df_fusion"]
df_fusion_cat= res["df_fusion_cat"]
df_ord_seg   = res["df_ord_seg"]
nombres_rfm  = res["nombres_rfm"]
nombres_dem  = res["nombres_dem"]
Frecuencia   = res["Frecuencia"]
Monetario    = res["Monetario"]
Recencia     = res["Recencia"]
met_gmm      = res["metricas_gmm"]
met_kp       = res["metricas_kp"]


# ══════════════════════════════════════════════
# SECCIÓN 1 — CONTEXTO DEL MERCADO
# ══════════════════════════════════════════════
if seccion == "🏠  Contexto del Mercado":
    st.markdown("""
    <div class="hero-box">
      <div class="hero-badge">Trabajo 2 · Marketing 2026-1 · UdeC</div>
      <h1>🛒 Segmentación de Clientes<br>E-commerce Global</h1>
      <p>Mercado 1 · Modelos GMM (RFM) + K-Prototypes (Demográfico) · Fusion Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("Contexto del Mercado", "¿Qué problema buscamos resolver?")
        st.markdown("""
        El dataset contiene registros transaccionales de un **E-commerce global** con miles de clientes
        distribuidos en múltiples países y categorías de producto.

        **El cliente** (la empresa que desea entrar al mercado) busca:
        - Identificar segmentos **más receptivos a promociones y descuentos**
        - Considerar la **categoría de producto preferida** como eje de personalización
        - Descubrir otros mercados meta relevantes más allá de la sensibilidad al precio
        """)
        st.markdown("""
        <div class="info-box">
        <b>🎯 Objetivo estratégico:</b> Desde la posición de una empresa entrante al mercado e-commerce,
        definir segmentos prioritarios y diseñar una estrategia de posicionamiento diferenciada para
        capturarlos eficientemente.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_clientes = df_customers.shape[0]
        total_ordenes  = df_orders.shape[0]
        n_paises       = df_customers["country"].nunique()
        n_categorias   = df_orders["category"].nunique()
        gasto_prom     = df_customers["avg_order_value_usd"].mean()

        for label, val in [
            ("Clientes registrados", f"{total_clientes:,}"),
            ("Órdenes totales",      f"{total_ordenes:,}"),
            ("Países representados", f"{n_paises}"),
            ("Categorías de producto", f"{n_categorias}"),
            ("Gasto promedio USD",   f"${gasto_prom:,.2f}"),
        ]:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:10px">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    section_header("Metodología", "Flujo del análisis")
    cols = st.columns(5)
    pasos = [
        ("1️⃣", "Carga y\nexploración",    "Revisar estructura, tipos y calidad de datos"),
        ("2️⃣", "Limpieza y\nwinsorización", "Filtros lógicos + tratamiento de outliers (1%–98%)"),
        ("3️⃣", "GMM\n(RFM)",              "Segmentación transaccional (k=4 clusters)"),
        ("4️⃣", "K-Prototypes\n(Demo.)",   "Segmentación demográfica mixta (k=3 clusters)"),
        ("5️⃣", "Fusión\n& estrategia",    "Cruce de modelos + recomendaciones de posicionamiento"),
    ]
    for col, (num, titulo, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="height: 200px; display: flex; flex-direction: column; justify-content: flex-start; align-items: center;">
              <div style="font-size:2.8rem; line-height: 1;">{num}</div>
              <div style="font-weight:700;font-size:1.05rem;margin:12px 0;white-space:pre-line">{titulo}</div>
              <div style="font-size:0.9rem;color:#666">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECCIÓN 2 — DESCRIPCIÓN DE DATOS
# ══════════════════════════════════════════════
elif seccion == "📋  Descripción de Datos":
    section_header("Descripción de Datos", "Exploración inicial de customers.csv y orders.csv")

    tab1, tab2 = st.tabs(["👤 Customers", "📦 Orders"])

    with tab1:
        st.markdown("### Vista previa")
        st.dataframe(df_customers.head(8), use_container_width=True)

        st.markdown("### Estadísticas descriptivas — Variables numéricas")
        st.dataframe(
            df_customers.select_dtypes(include="number").describe().T.round(2),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribución de Membership Tier**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_customers["membership_tier"].value_counts().plot(
                kind="bar", ax=ax, color=["#0a2342", "#1a6fb5", "#f5a623", "#e74c3c"], edgecolor="white"
            )
            ax.set_xlabel("Tier"); ax.set_ylabel("N° clientes"); ax.set_title("Membership Tier")
            plt.xticks(rotation=0); plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**Distribución de Género**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_customers["gender"].value_counts().plot(
                kind="bar", ax=ax, color=["#9b59b6", "#3498db", "#95a5a6"], edgecolor="white"
            )
            ax.set_xlabel("Género"); ax.set_ylabel("N° clientes"); ax.set_title("Género")
            plt.xticks(rotation=0); plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("**Top 10 Países**")
        top_paises = df_customers["country"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 3))
        top_paises.plot(kind="barh", ax=ax, color="#1a6fb5")
        ax.set_xlabel("N° clientes"); ax.invert_yaxis(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("### Vista previa")
        st.dataframe(df_orders.head(8), use_container_width=True)

        st.markdown("### Estadísticas descriptivas — Variables numéricas")
        st.dataframe(
            df_orders.select_dtypes(include="number").describe().T.round(2),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ventas por Categoría**")
            cat_ventas = df_orders.groupby("category")["total_amount_usd"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cat_ventas.plot(kind="bar", ax=ax, color="#0a2342", edgecolor="white")
            ax.set_xlabel("Categoría"); ax.set_ylabel("Total USD"); ax.set_title("Revenue por Categoría")
            plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**Distribución de Descuentos (%)**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_orders["discount_pct"].value_counts().sort_index().plot(
                kind="bar", ax=ax, color="#f5a623", edgecolor="white"
            )
            ax.set_xlabel("Descuento (%)"); ax.set_ylabel("N° órdenes"); ax.set_title("Frecuencia de Descuentos")
            plt.xticks(rotation=0); plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("**Nulos por columna**")
        nulos = df_orders.isnull().sum()
        nulos = nulos[nulos > 0]
        if nulos.empty:
            st.success("✅ No hay valores nulos en orders.csv")
        else:
            st.dataframe(nulos.rename("Nulos"), use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 3 — LIMPIEZA DE DATOS
# ══════════════════════════════════════════════
elif seccion == "🧹  Limpieza de Datos":
    section_header("Limpieza de Datos", "Filtros lógicos y tratamiento de outliers")

    st.markdown("""
    <div class="info-box">
    El dataset es <b>data original sin procesar</b>, por lo que se aplicaron los siguientes pasos
    antes de cualquier modelado:
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔍 Filtros Aplicados")
        filtros = {
            "customer_id en orders": (df_customers["customer_id"].isin(df_orders["customer_id"])).sum(),
            "total_orders > 0":       (df_customers["total_orders"] > 0).sum(),
            "avg_order_value > 0":    (df_customers["avg_order_value_usd"] > 0).sum(),
            "days_since_purchase ≥ 0":(df_customers["days_since_last_purchase"] >= 0).sum(),
        }
        df_filtros = pd.DataFrame(
            {"Criterio": list(filtros.keys()), "Clientes que cumplen": list(filtros.values())}
        )
        st.dataframe(df_filtros, use_container_width=True, hide_index=True)

        st.markdown("#### ✂️ Winsorización (1% – 98%)")
        st.markdown("""
        Para estabilizar los clusters y reducir el efecto de valores extremos,
        se aplicó **winsorización** a las tres variables RFM:
        - `total_orders` (Frecuencia)
        - `avg_order_value_usd` (Monetario)
        - `days_since_last_purchase` (Recencia)
        """)

    with col2:
        st.markdown(f"#### 📏 Clientes Finales para el Modelo")
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:16px">
          <div class="kpi-value">{df_RFM.shape[0]:,}</div>
          <div class="kpi-label">Clientes tras limpieza</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value">{df_customers.shape[0] - df_RFM.shape[0]:,}</div>
          <div class="kpi-label">Registros eliminados</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📦 Boxplots RFM — Post Winsorización")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate([Frecuencia, Monetario, Recencia]):
        sns.boxplot(y=df_RFM[col], ax=axes[i], color="#3498db")
        axes[i].set_title(f"Boxplot: {col}", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 📊 Estadísticas RFM — Dataset Limpio")
    st.dataframe(df_RFM[[Frecuencia, Monetario, Recencia]].describe().T.round(2), use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 4 — GMM
# ══════════════════════════════════════════════
elif seccion == "📈  Modelo RFM — GMM":
    section_header("Modelo RFM — Gaussian Mixture Model (GMM)", "Segmentación transaccional con variables continuas")

    st.markdown("""
    <div class="info-box">
    <b>¿Por qué GMM?</b> Las variables RFM son continuas y pueden presentar distribuciones
    superpuestas. GMM permite asignación probabilística de clientes a segmentos, capturando
    mejor la incertidumbre en los límites entre grupos.
    </div>
    """, unsafe_allow_html=True)

    tab_sel, tab_modelo, tab_viz = st.tabs(
        ["📊 Selección de k", "📋 Perfiles de Clusters", "🌐 Visualización 3D"]
    )

    with tab_sel:
        st.markdown("#### Métricas de Selección (k = 2 a 10)")
        fig, axes = plt.subplots(3, 2, figsize=(14, 13))
        pares = [
            ("BIC",      "Criterio BIC vs AIC (menor = mejor)",      "#d95f02"),
            ("Silueta",  "Coeficiente de Silueta (mayor = mejor)",    "#7570b3"),
            ("DBI",      "Davies-Bouldin (menor = mejor)",            "#d95f02"),
            ("CHI",      "Calinski-Harabasz (mayor = mejor)",         "#7570b3"),
            ("Entropia", "Entropía Escalada (cercana a 1 = mejor)",   "#d95f02"),
            ("LogLik",   "Log-Likelihood (mayor = mejor)",            "#7570b3"),
        ]
        for ax, (metrica, titulo, color) in zip(axes.flatten(), pares):
            sns.lineplot(x="k", y=metrica, data=met_gmm, ax=ax, marker="o", color=color)
            if metrica == "BIC":
                sns.lineplot(x="k", y="AIC", data=met_gmm, ax=ax, marker="s", color="#2ca02c", label="AIC")
                ax.legend()
            ax.set_title(titulo, fontweight="bold", fontsize=10)
            ax.set_xlabel("k"); ax.set_ylabel("Valor"); ax.grid(True, ls="--", alpha=0.4)
            ax.axvline(4, color="red", ls="--", alpha=0.5, label="k=4 seleccionado")
        fig.suptitle("Evaluación de Métricas GMM", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="warn-box">
        <b>🏆 Decisión: k = 4</b><br>
        La combinación de métricas indica un punto de inflexión claro en k=4:
        BIC/AIC comienzan a estabilizarse, la Silueta muestra un buen balance y
        la Entropía confirma separación adecuada entre perfiles.
        </div>
        """, unsafe_allow_html=True)

    with tab_modelo:
        st.markdown("#### Perfiles Promedio por Cluster GMM (k=4)")
        df_perf_gmm = df_GMM.groupby("Cluster")[[Frecuencia, Monetario, Recencia]].mean().round(2)
        df_perf_gmm["Nombre"] = df_perf_gmm.index.map(nombres_rfm)
        df_perf_gmm["N clientes"] = df_GMM["Cluster"].value_counts().sort_index()
        st.dataframe(df_perf_gmm.rename(columns={
            Frecuencia: "Frecuencia (Órdenes)",
            Monetario:  "Monetario (Avg USD)",
            Recencia:   "Recencia (Días)",
        }), use_container_width=True)

        st.markdown("#### Distribución de clientes por Cluster")
        fig, ax = plt.subplots(figsize=(8, 4))
        sizes = df_GMM["Cluster"].value_counts().sort_index()
        labels = [nombres_rfm[k] for k in sizes.index]
        ax.bar(labels, sizes.values, color=CLUSTER_COLORS_RFM[:4], edgecolor="white")
        for i, v in enumerate(sizes.values):
            ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
        ax.set_ylabel("N° Clientes"); ax.set_title("Tamaño de Clusters GMM")
        plt.xticks(rotation=15, ha="right"); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("#### Descripción de Segmentos")
        descripciones = {
            "Clientes Premium":                    ("🏆", "#f39c12", "Alto gasto promedio por orden. Compradores de alto valor que priorizan calidad sobre precio."),
            "Clientes Históricos/Transaccionales": ("🔄", "#2ecc71", "Alta frecuencia de compra. Clientes leales y recurrentes, core del negocio."),
            "Cliente Promedio":                    ("📊", "#3498db", "Comportamiento equilibrado. Actividad reciente, valores medios en todas las métricas RFM."),
            "Cliente Ocasional":                   ("💤", "#e74c3c", "Baja frecuencia, menor gasto. Mayor recencia (más días sin comprar). Alta probabilidad de churn."),
        }
        for nombre, (icono, color, desc) in descripciones.items():
            st.markdown(f"""
            <div class="segment-card" style="border-color:{color}; background-color: #fdfdfd; padding: 16px;">
              <b style="color: #0a2342; font-size: 1.15rem; display: inline-block; margin-bottom: 6px;">
                {icono} {nombre}
              </b>
              <br>
              <span style="font-size: 0.92rem; color: #333333; line-height: 1.4;">
                {desc}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with tab_viz:
        st.markdown("#### Visualización 3D Interactiva — Espacio RFM")
        df_plot = df_GMM.copy()
        df_plot["Nombre Cluster"] = df_plot["Cluster"].map(nombres_rfm)
        fig_3d = px.scatter_3d(
            df_plot, x=Recencia, y=Frecuencia, z=Monetario,
            color="Nombre Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Segmentación RFM — GMM (k=4)",
            opacity=0.55, height=650,
        )
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis_title="Recencia (Días)",
                yaxis_title="Frecuencia (Órdenes)",
                zaxis_title="Monetario (USD)",
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 5 — K-PROTOTYPES
# ══════════════════════════════════════════════
elif seccion == "👥  Modelo Demográfico — K-Prototypes":
    section_header("Modelo Demográfico — K-Prototypes", "Segmentación con datos mixtos: edad (continua) + género y membresía (categóricas)")

    st.markdown("""
    <div class="info-box">
    <b>¿Por qué K-Prototypes?</b> A diferencia de K-Means (solo variables continuas) o K-Modes
    (solo categóricas), K-Prototypes maneja naturalmente datos <b>mixtos</b>, combinando
    distancia euclidiana para edad y distancia de Hamming para género y membresía.
    </div>
    """, unsafe_allow_html=True)

    tab_sel, tab_modelo, tab_viz = st.tabs(
        ["📊 Selección de k", "📋 Perfiles de Clusters", "🌐 Visualización 3D"]
    )

    with tab_sel:
        st.markdown("#### Métricas de Selección K-Prototypes (k = 2 a 10)")
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        pares = [
            ("Costo",   "Costo Total / Codo (menor = mejor)",    "#d95f02"),
            ("Silueta", "Coeficiente de Silueta (mayor = mejor)","#7570b3"),
            ("DBI",     "Davies-Bouldin (menor = mejor)",        "#d95f02"),
            ("CHI",     "Calinski-Harabasz (mayor = mejor)",     "#7570b3"),
        ]
        for ax, (metrica, titulo, color) in zip(axes.flatten(), pares):
            sns.lineplot(x="k", y=metrica, data=met_kp, ax=ax, marker="o", color=color)
            ax.set_title(titulo, fontweight="bold", fontsize=10)
            ax.set_xlabel("k"); ax.set_ylabel("Valor"); ax.grid(True, ls="--", alpha=0.4)
            ax.axvline(3, color="red", ls="--", alpha=0.5, label="k=3 seleccionado")
            ax.legend()
        fig.suptitle("Evaluación de Métricas K-Prototypes", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="warn-box">
        <b>🏆 Decisión: k = 3</b><br>
        El codo en la curva de costo aparece claramente en k=3. La Silueta y CH
        también muestran un balance adecuado en 3 clusters, que además produce
        perfiles demográficos interpretables y accionables.
        </div>
        """, unsafe_allow_html=True)

    with tab_modelo:
        st.markdown("#### Perfiles por Cluster K-Prototypes (k=3)")
        dic_gender_rev = {0: "Female", 1: "Male", 2: "Other"}
        dic_tier_rev   = {0: "Free",   1: "Silver", 2: "Gold", 3: "Platinum"}

        df_perf_kp = df_dem_final.groupby("Cluster").agg(
            Edad_Promedio=("age", "mean"),
            Genero_Modal =("gender", lambda x: dic_gender_rev.get(x.mode()[0], x.mode()[0])),
            Membresia_Modal=("membership_tier", lambda x: dic_tier_rev.get(x.mode()[0], x.mode()[0])),
            N_Clientes  =("age", "count"),
        ).round(2)
        df_perf_kp["Nombre"] = df_perf_kp.index.map(nombres_dem)
        st.dataframe(df_perf_kp, use_container_width=True)

        st.markdown("#### Descripción de Segmentos Demográficos")
        descripciones_dem = {
            "Hombres Más Jóvenes":                    ("🧑", "#3498db", "Edad promedio más baja (~30 años). Hombres predominantes, membresía predominantemente Free/Silver. Alta adopción digital."),
            "Mujeres Poder Adq. Medio-Bajo":          ("👩", "#9b59b6", "Edad intermedia (~35 años). Predominancia femenina, membresía media. Potencial de upselling con incentivos correctos."),
            "Hombres Mayor Edad Poder Adq. Medio":    ("👨", "#e67e22", "Mayor edad promedio (~47-48 años). Membresía más alta (Gold/Platinum). Clientes consolidados con mayor poder adquisitivo."),
        }
        for nombre, (icono, color, desc) in descripciones_dem.items():
            st.markdown(f"""
            <div class="segment-card" style="border-color:{color}; background-color: #fdfdfd; padding: 16px;">
              <b style="color: #0a2342; font-size: 1.15rem; display: inline-block; margin-bottom: 6px;">
                {icono} {nombre}
              </b>
              <br>
              <span style="font-size: 0.92rem; color: #333333; line-height: 1.4;">
                {desc}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with tab_viz:
        st.markdown("#### Visualización 3D — Nube con Jitter (Datos Mixtos)")
        np.random.seed(42)
        df_plot_kp = df_dem_final.copy()
        df_plot_kp["Nombre Cluster"] = df_plot_kp["Cluster"].map(nombres_dem)
        df_plot_kp["gender_j"]      = df_plot_kp["gender"]          + np.random.uniform(-0.2, 0.2, len(df_plot_kp))
        df_plot_kp["membership_j"]  = df_plot_kp["membership_tier"] + np.random.uniform(-0.2, 0.2, len(df_plot_kp))

        fig_3d_kp = px.scatter_3d(
            df_plot_kp, x="age", y="gender_j", z="membership_j",
            color="Nombre Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Segmentación Demográfica — K-Prototypes (k=3)",
            opacity=0.55, height=650,
        )
        fig_3d_kp.update_traces(marker=dict(size=3))
        fig_3d_kp.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis_title="Edad (años)",
                yaxis_title="Género",
                yaxis=dict(tickvals=[0, 1, 2], ticktext=["Female", "Male", "Other"]),
                zaxis_title="Membresía",
                zaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Free", "Silver", "Gold", "Platinum"]),
            ),
        )
        st.plotly_chart(fig_3d_kp, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 6 — ANÁLISIS DE FUSIÓN
# ══════════════════════════════════════════════
elif seccion == "🔗  Análisis de Fusión":
    section_header("Análisis de Fusión", "Cruce de segmentos RFM (GMM) × Demográfico (K-Prototypes)")

    st.markdown("""
    <div class="info-box">
    La fusión combina los <b>4 clusters RFM</b> × <b>3 clusters demográficos</b>,
    generando <b>12 micro-segmentos</b> únicos. Cada micro-segmento revela un perfil
    completo: comportamiento de compra + demografía.
    </div>
    """, unsafe_allow_html=True)

    tab_mat, tab_heat, tab_burbuja, tab_descuento, tab_categoria = st.tabs([
        "📊 Matriz de Fusión",
        "🌡️ Heatmaps Variables",
        "🫧 Mapa de Burbujas",
        "💰 Análisis de Descuentos",
        "🛍️ Preferencia de Categoría",
    ])

    with tab_mat:
        st.markdown("#### Distribución de Clientes por Micro-Segmento")
        # Mapear nombres
        df_fus_named = df_fusion.copy()
        df_fus_named["Nombre_RFM"] = df_fus_named["Cluster_RFM"].map(nombres_rfm)
        df_fus_named["Nombre_DEM"] = df_fus_named["Cluster_DEM"].map(nombres_dem)

        tabla = pd.crosstab(df_fus_named["Nombre_RFM"], df_fus_named["Nombre_DEM"])
        tabla_pct = tabla.div(tabla.sum().sum()) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            heatmap_anotado(tabla, ax, fmt="d", cmap="YlOrRd",
                            title="Clientes por Micro-Segmento (Absoluto)",
                            xlabel="Cluster Demográfico", ylabel="Cluster RFM")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            heatmap_anotado(tabla_pct, ax, fmt=".1f", cmap="YlOrRd",
                            title="% de Clientes por Micro-Segmento",
                            xlabel="Cluster Demográfico", ylabel="Cluster RFM")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Tabla detallada de Micro-Segmentos")
        ms_counts = df_fusion["Micro_Segmento"].value_counts().reset_index()
        ms_counts.columns = ["Micro-Segmento", "N° Clientes"]
        ms_counts["% del Total"] = (ms_counts["N° Clientes"] / ms_counts["N° Clientes"].sum() * 100).round(1)
        st.dataframe(ms_counts, use_container_width=True, hide_index=True)

    with tab_heat:
        st.markdown("#### Heatmaps de Variables Clave por Micro-Segmento")
        df_fus_n = df_fusion.copy()
        df_fus_n["Nombre_RFM"] = df_fus_n["Cluster_RFM"].map(nombres_rfm)
        df_fus_n["Nombre_DEM"] = df_fus_n["Cluster_DEM"].map(nombres_dem)

        vars_heat = [
            ("discount_pct_prom",     "% Descuento Promedio",        "RdYlGn"),
            ("session_duration_prom", "Duración Sesión (min)",        "Blues"),
            ("pages_viewed_prom",     "Páginas Vistas",               "Purples"),
            ("age",                   "Edad Media",                   "Oranges"),
            ("returns_made",          "Devoluciones",                 "Reds"),
            ("rating_prom",           "Rating Promedio",              "RdYlGn"),
            ("is_repeat_pct",         "% Clientes Recurrentes",       "Greens"),
            ("churned",               "% Churn",                      "RdYlGn_r"),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        for (col, lbl, cmap), ax in zip(vars_heat, axes.flatten()):
            pivot = df_fus_n.groupby(["Nombre_RFM", "Nombre_DEM"])[col].mean().unstack()
            heatmap_anotado(pivot, ax, fmt=".2f", cmap=cmap, title=lbl,
                            xlabel="Cluster DEM", ylabel="Cluster RFM")
        fig.suptitle("Heatmaps de Variables por Micro-Segmento", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab_burbuja:
        st.markdown("#### Mapa de Burbujas: Frecuencia vs. Descuento Promedio")
        burbuja = df_fusion.groupby("Micro_Segmento").agg(
            descuento  =("discount_pct_prom", "mean"),
            frecuencia =("n_ordenes",          "mean"),
            n_clientes =("age",                "count"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(13, 7))
        colores = plt.cm.tab20(np.linspace(0, 1, len(burbuja)))
        scatter = ax.scatter(
            burbuja["descuento"], burbuja["frecuencia"],
            s=burbuja["n_clientes"] * 3,
            c=colores, alpha=0.75, edgecolors="black", linewidths=1.2,
        )
        for i, row in burbuja.iterrows():
            ax.text(row["descuento"], row["frecuencia"] + 0.04,
                    f"{row['Micro_Segmento'].split(' | ')[0][:20]}…\n(N={row['n_clientes']})",
                    ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("Descuento Promedio (%)"); ax.set_ylabel("Frecuencia Promedio (N° Órdenes)")
        ax.set_title("F4: Frecuencia de Compras vs. Descuento Promedio\n(Tamaño burbuja = N° clientes)",
                     fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="info-box">
        Los segmentos ubicados en la esquina <b>superior derecha</b> (alta frecuencia + alto descuento)
        son los más <b>receptivos a promociones</b>. Los de la esquina superior izquierda compran
        frecuentemente <b>sin necesitar descuento</b> — mayor margen potencial.
        </div>
        """, unsafe_allow_html=True)

    with tab_descuento:
        st.markdown("#### Distribución de Transacciones por Nivel de Descuento")
        tabla_desc = pd.crosstab(df_ord_seg["discount_pct"], df_ord_seg["Micro_Segmento"])
        tabla_desc_pct = tabla_desc.div(tabla_desc.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        tabla_desc.plot(kind="bar", stacked=True, colormap="tab20", edgecolor="white", width=0.7, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("Volumen de Compras por Descuento (Log Scale)", fontweight="bold")
        axes[0].set_xlabel("Descuento (%)"); axes[0].set_ylabel("N° Compras (log)")
        axes[0].legend(title="Micro-Segmento", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)

        tabla_desc_pct.plot(kind="bar", stacked=True, colormap="tab20", edgecolor="white", width=0.7, ax=axes[1])
        axes[1].set_title("Distribución Porcentual (100% Normalizado)", fontweight="bold")
        axes[1].set_xlabel("Descuento (%)"); axes[1].set_ylabel("% Compras")
        axes[1].get_legend().remove()

        for x_idx, (_, row) in enumerate(tabla_desc_pct.iterrows()):
            bottom = 0.0
            for _, val in row.items():
                if val > 4.0:
                    y_pos = bottom + val / 2.0
                    txt = axes[1].text(x_idx, y_pos, f"{val:.1f}%",
                                       ha="center", va="center", fontsize=6.5, fontweight="bold", color="black")
                    txt.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="white"),
                                          path_effects.Normal()])
                bottom += val
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab_categoria:
        st.markdown("#### Preferencia de Categorías por Micro-Segmento")
        tabla_cat = pd.crosstab(df_ord_seg["category"], df_ord_seg["Micro_Segmento"])
        tabla_cat_pct = tabla_cat.div(tabla_cat.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        tabla_cat.plot(kind="barh", stacked=True, colormap="tab20", edgecolor="white", width=0.6, ax=axes[0])
        axes[0].set_title("Volumen Absoluto por Categoría", fontweight="bold")
        axes[0].set_xlabel("N° Compras")
        axes[0].legend(title="Micro-Segmento", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)

        tabla_cat_pct.plot(kind="barh", stacked=True, colormap="tab20", edgecolor="white", width=0.6, ax=axes[1])
        axes[1].set_title("Distribución Porcentual (100%)", fontweight="bold")
        axes[1].set_xlabel("% de Compras"); axes[1].get_legend().remove()

        for y_idx, (_, row) in enumerate(tabla_cat_pct.iterrows()):
            bottom = 0.0
            for _, val in row.items():
                if val > 4.0:
                    x_pos = bottom + val / 2.0
                    txt = axes[1].text(x_pos, y_idx, f"{val:.1f}%",
                                       ha="center", va="center", fontsize=6.5, fontweight="bold", color="black")
                    txt.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="white"),
                                          path_effects.Normal()])
                bottom += val
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
# SECCIÓN 7 — CONCLUSIONES Y POSICIONAMIENTO
# ══════════════════════════════════════════════
elif seccion == "🎯  Conclusiones y Posicionamiento":
    section_header("Conclusiones y Estrategia de Posicionamiento",
                   "Mercados meta relevantes y recomendaciones para la empresa entrante")

    st.markdown("### 🏆 Mercados Meta Seleccionados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="segment-card" style="border-color:#f39c12;background:#fffbf0;color:#1a1a1a">
          <b style="color:#1a1a1a">🥇 Mercado Meta 1 — Clientes Históricos/Transaccionales × Hombres Más Jóvenes</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Mayor frecuencia de compra + alta receptividad a descuentos.
          Son el segmento más activo y con mayor volumen transaccional.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Electronics, Sports & Outdoors, Clothing<br>
          <b style="color:#1a1a1a">Estrategia:</b> Programa de lealtad con descuentos escalonados, push notifications,
          experiencia mobile-first, early access a promociones flash.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="segment-card" style="border-color:#e74c3c;background:#fff5f5;color:#1a1a1a">
          <b style="color:#1a1a1a">🥈 Mercado Meta 2 — Clientes Premium × Hombres Mayor Edad</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Mayor gasto promedio + membresía Gold/Platinum + menor sensibilidad
          al precio. Representan el mayor valor monetario por cliente.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Electronics, Home & Garden, Books<br>
          <b style="color:#1a1a1a">Estrategia:</b> Posicionamiento premium, envío express gratuito, servicio al cliente
          prioritario, recomendaciones personalizadas de alta gama.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="segment-card" style="border-color:#9b59b6;background:#faf5ff;color:#1a1a1a">
          <b style="color:#1a1a1a">🥉 Mercado Meta 3 — Clientes Promedio × Mujeres Poder Adq. Medio-Bajo</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Gran volumen potencial de clientes. Alta receptividad a descuentos
          y promociones. Segmento con mayor potencial de migración a Premium.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Beauty & Health, Clothing, Home<br>
          <b style="color:#1a1a1a">Estrategia:</b> Campañas estacionales, cupones de descuento, membership upgrade
          incentivado, contenido inspiracional vía newsletter.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="segment-card" style="border-color:#95a5a6;background:#f8f9fa;color:#1a1a1a">
          <b style="color:#1a1a1a">⚠️ Segmento de Atención — Clientes Ocasionales</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Alta probabilidad de churn. Requieren estrategias de reactivación
          antes de que abandonen definitivamente.<br><br>
          <b style="color:#1a1a1a">Estrategia:</b> Win-back campaigns, descuentos de reactivación, recordatorios
          de wishlist, encuestas de satisfacción para entender la fricción.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🗺️ Estrategia de Posicionamiento")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🎯 Propuesta de Valor**
        - Plataforma de descuentos inteligentes y personalizados
        - Categorías de alta demanda: Electronics y Sports
        - Programa de membresía con beneficios tangibles
        - Experiencia mobile-first para segmento joven
        """)
    with col_b:
        st.markdown("""
        **📣 Canales de Adquisición**
        - Social media / influencers para segmento joven
        - Email marketing segmentado por tier de membresía
        - Retargeting para clientes ocasionales
        - SEO de nicho en categorías específicas
        """)
    with col_c:
        st.markdown("""
        **💡 Diferenciación**
        - Algoritmo de descuento personalizado por segmento
        - Transparencia en precios sin letra chica
        - Envío express para miembros Premium
        - Sistema de reviews verificados (confiar en ratings)
        """)

    st.divider()
    st.markdown("### 📊 Resumen Ejecutivo de Métricas de los Modelos")
    col1, col2, col3, col4 = st.columns(4)
    best_gmm = met_gmm[met_gmm["k"] == 4].iloc[0]
    best_kp  = met_kp[met_kp["k"]   == 3].iloc[0]
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">4</div><div class="kpi-label">Clusters GMM (RFM)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_gmm["Silueta"]:.3f}</div><div class="kpi-label">Silueta GMM</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">3</div><div class="kpi-label">Clusters K-Proto (Demo.)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_kp["Silueta"]:.3f}</div><div class="kpi-label">Silueta K-Prototypes</div></div>', unsafe_allow_html=True)
