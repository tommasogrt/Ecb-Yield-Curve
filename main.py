"""
Euro Area Sovereign Yield Curve Modeling, PCA Decomposition, and Stress Testing
-------------------------------------------------------------------------------
Author: Data Science Candidate for ECB Traineeship
Data Source: European Central Bank (ECB) Data Portal API
Description: Parametrization of the Euro Area AAA spot yield curve using the
             Nelson-Siegel model, dimensionality reduction via Principal
             Component Analysis (PCA), and macroeconomic stress testing.
"""

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA


# =============================================================================
# 1. DATA EXTRACTION (ECB API)
# =============================================================================
def fetch_ecb_yield_data(start_date: str = "2020-01-01") -> pd.DataFrame:
    """Scarica i dati storici della struttura a termine dei tassi spot dell'Eurozona

    dall'API REST ufficiale del servizio dati della BCE (ECB Data Portal).
    """
    print("[INFO] Download serie storiche dalla BCE in corso...")
    url = (
        f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.?"
        f"startPeriod={start_date}&format=csvdata"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        df = df[["TIME_PERIOD", "DATA_TYPE_FM", "OBS_VALUE"]]
        df.columns = ["Date", "Maturity_Code", "Yield"]

        # Mappatura delle scadenze espresse in frazioni di anno
        maturity_map = {
            "B03M": 0.25,
            "B06M": 0.5,
            "SR_1Y": 1.0,
            "SR_2Y": 2.0,
            "SR_3Y": 3.0,
            "SR_5Y": 5.0,
            "SR_7Y": 7.0,
            "SR_10Y": 10.0,
            "SR_15Y": 15.0,
            "SR_30Y": 30.0,
        }

        df = df[df["Maturity_Code"].isin(maturity_map.keys())].copy()
        df["Maturity"] = df["Maturity_Code"].map(maturity_map)

        # Riorganizzazione in matrice temporale (Date x Scadenze)
        df_pivot = df.pivot(
            index="Date", columns="Maturity", values="Yield"
        ).dropna()
        df_pivot.index = pd.to_datetime(df_pivot.index)
        df_pivot = df_pivot.sort_index()

        print(
            f"[SUCCESS] Dati acquisiti: {df_pivot.shape[0]} osservazioni dal "
            f"{df_pivot.index.min().strftime('%Y-%m-%d')} al {df_pivot.index.max().strftime('%Y-%m-%d')}."
        )
        return df_pivot

    except Exception as err:
        print(f"[ERROR] Impossibile recuperare i dati dall'API BCE: {err}")
        raise err


# =============================================================================
# 2. QUANTITATIVE MODELING (NELSON-SIEGEL)
# =============================================================================
class NelsonSiegelModel:
    """Modello parametrico di Nelson-Siegel (1987) per la stima e l'interpolazione

    della curva dei rendimenti a termine.

    Parametri:
        beta0 (float): Livello dei tassi a lungo termine (Level).
        beta1 (float): Inclinazione/pendenza della curva (Slope).
        beta2 (float): Curvatura della struttura a termine (Curvature).
        tau0  (float): Fattore di scala temporale della curvatura.
    """

    def __init__(self, tau0: float = 2.0):
        self.tau0 = tau0

    def predict(
        self,
        maturities: np.ndarray,
        beta0: float,
        beta1: float,
        beta2: float,
    ) -> np.ndarray:
        """Calcola i tassi teorici interpolati per ciascuna scadenza fornita."""
        maturities = np.array(maturities)
        factor1 = (1 - np.exp(-maturities / self.tau0)) / (
            maturities / self.tau0
        )
        factor2 = factor1 - np.exp(-maturities / self.tau0)
        return beta0 + beta1 * factor1 + beta2 * factor2

    def fit_curve(self, maturities: np.ndarray, yields: np.ndarray):
        """Stima i parametri ottimali (beta0, beta1, beta2) minimizzando la somma

        degli scarti quadratici (MSE) rispetto ai tassi osservati sul mercato.
        """
        maturities = np.array(maturities)
        yields = np.array(yields)

        def loss_function(betas):
            b0, b1, b2 = betas
            predictions = self.predict(maturities, b0, b1, b2)
            return np.sum((yields - predictions) ** 2)

        # Inizializzazione basata sull'osservazione dei punti estremi della curva
        initial_guess = [yields[-1], yields[0] - yields[-1], 0.0]
        optimization_result = minimize(
            loss_function, initial_guess, method="Nelder-Mead"
        )
        return optimization_result.x


# =============================================================================
# 3. MAIN EXECUTION & WORKFLOW
# =============================================================================
if __name__ == "__main__":
    # Estrazione Dati
    yield_data = fetch_ecb_yield_data()
    maturities = yield_data.columns.values
    ns_model = NelsonSiegelModel(tau0=2.0)

    # -------------------------------------------------------------------------
    # Fitting Ultima Osservazione
    # -------------------------------------------------------------------------
    latest_yields = yield_data.iloc[-1].values
    b0_latest, b1_latest, b2_latest = ns_model.fit_curve(
        maturities, latest_yields
    )

    print("\n" + "=" * 55)
    print(f" NELSON-SIEGEL PARAMETERS ({yield_data.index[-1].date()})")
    print("=" * 55)
    print(f" Beta 0 (Level - Lungo Termine):  {b0_latest:8.4f}%")
    print(f" Beta 1 (Slope - Pendenza):        {b1_latest:8.4f}%")
    print(f" Beta 2 (Curvature - Gobba):       {b2_latest:8.4f}%")

    # -------------------------------------------------------------------------
    # Decomposizione Fattoriale tramite PCA
    # -------------------------------------------------------------------------
    pca = PCA(n_components=3)
    pca.fit(yield_data)
    variance_ratios = pca.explained_variance_ratio_

    print("\n" + "=" * 55)
    print(" PRINCIPAL COMPONENT ANALYSIS (PCA) RESULTS")
    print("=" * 55)
    print(f" PC1 (Level Variance):       {variance_ratios[0]*100:6.2f}%")
    print(f" PC2 (Slope Variance):       {variance_ratios[1]*100:6.2f}%")
    print(f" PC3 (Curvature Variance):   {variance_ratios[2]*100:6.2f}%")
    print(f" Total Variance Explained:   {sum(variance_ratios)*100:6.2f}%")
    print("=" * 55 + "\n")

    # -------------------------------------------------------------------------
    # Simulazione di Stress Testing
    # -------------------------------------------------------------------------
    dense_maturities = np.linspace(0.25, 30, 100)
    base_curve = ns_model.predict(
        dense_maturities, b0_latest, b1_latest, b2_latest
    )

    # Scenario 1: Parallel Hawkish Shift (+100 bps)
    hawkish_curve = base_curve + 1.0

    # Scenario 2: Bear Flattening (Tassi a breve +150 bps, a lungo +25 bps)
    flattening_vector = np.linspace(1.5, 0.25, len(maturities))
    flattening_yields = latest_yields + flattening_vector
    b0_fl, b1_fl, b2_fl = ns_model.fit_curve(maturities, flattening_yields)
    flattening_curve = ns_model.predict(
        dense_maturities, b0_fl, b1_fl, b2_fl
    )

    # -------------------------------------------------------------------------
    # Visualizzazione Grafica
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Grafico 1: Fitting Attuale vs Stress Test
    axes[0].scatter(
        maturities,
        latest_yields,
        color="crimson",
        label="ECB Spot Rates (Observed)",
        zorder=5,
    )
    axes[0].plot(
        dense_maturities,
        base_curve,
        label="Nelson-Siegel Fit",
        color="navy",
        linewidth=2,
    )
    axes[0].plot(
        dense_maturities,
        hawkish_curve,
        label="Scenario 1: Hawkish Shift (+100bps)",
        color="darkred",
        linestyle="--",
    )
    axes[0].plot(
        dense_maturities,
        flattening_curve,
        label="Scenario 2: Bear Flattening",
        color="darkorange",
        linestyle=":",
    )
    axes[0].set_title(
        f'Euro Area Yield Curve & Stress Scenarios ({yield_data.index[-1].strftime("%Y-%m-%d")})'
    )
    axes[0].set_xlabel("Maturity (Years)")
    axes[0].set_ylabel("Yield Spot Rate (%)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Grafico 2: Evoluzione Storica dei Fattori (Serie Storica 2020-Oggi)
    history_betas = [
        ns_model.fit_curve(maturities, row.values)
        for _, row in yield_data.iterrows()
    ]
    df_betas = pd.DataFrame(
        history_betas,
        index=yield_data.index,
        columns=["Level", "Slope", "Curvature"],
    )

    axes[1].plot(
        df_betas.index,
        df_betas["Level"],
        label="Level (beta0)",
        color="navy",
    )
    axes[1].plot(
        df_betas.index,
        df_betas["Slope"],
        label="Slope (beta1)",
        color="crimson",
    )
    axes[1].plot(
        df_betas.index,
        df_betas["Curvature"],
        label="Curvature (beta2)",
        color="forestgreen",
    )
    axes[1].set_title("Historical Dynamics of Nelson-Siegel Factors")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Parameter Value (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.show()