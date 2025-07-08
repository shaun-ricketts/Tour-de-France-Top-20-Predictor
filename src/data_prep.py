"""data_prep.py
Preprocess cycling race datasets to build a feature table for a Tour‑de‑France prediction model.

Public API
----------
preprocess_tdf_data(folder_path: str, output_path: str | None = None) -> pd.DataFrame

Usage example
-------------
>>> from data_prep import preprocess_tdf_data
>>> df = preprocess_tdf_data(r"D:/Data/Cycling/TDF_Predictor", "tdf_prepared_2011_2024.csv")
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def _load_raw(folder_path: str) -> dict[str, pd.DataFrame]:
    """Read every raw CSV file required for feature engineering."""

    join = os.path.join
    return {
        "uwt_res": pd.read_csv(
            join(folder_path, "UWT_Race_Results.csv"),
            usecols=["Pos", "Born", "Rider URL", "YEAR", "RACE", "RACE_URL"],
        ),
        "uwt_races": pd.read_csv(join(folder_path, "UWT_Races.csv")),
        "pt_res": pd.read_csv(
            join(folder_path, "PT_Race_Results.csv"),
            usecols=["Pos", "Born", "Rider URL", "YEAR", "RACE", "RACE_URL"],
        ),
        "pt_races": pd.read_csv(join(folder_path, "PT_Races.csv")),
        "fc_rank": pd.read_csv(
            join(folder_path, "fc_rankings.csv"),
            usecols=["rider_id", "Rider", "Year", "Points"],
        ),
        "gt_history": pd.read_csv(join(folder_path, "GT_History.csv")),
    }

# ---------------------------------------------------------------------
# Grand‑Tour feature engineering
# ---------------------------------------------------------------------

def _prepare_gt_history(gt_history: pd.DataFrame) -> pd.DataFrame:
    """Return GT‑specific yearly features for Tour‑de‑France participants."""

    gt_history = gt_history.copy()
    gt_history["Pos_clean"] = pd.to_numeric(gt_history["Pos"], errors="coerce")

    def _race_order(row) -> int:
        if row["Year"] == 2020:  # pandemic‑shifted calendar
            mapping = {"Tour de France": 1, "Giro d'Italia": 2, "Vuelta a España": 3}
        else:
            mapping = {"Giro d'Italia": 1, "Tour de France": 2, "Vuelta a España": 3}
        return mapping.get(row["Race"], 99)

    gt_history["race_order"] = gt_history.apply(_race_order, axis=1)
    gt_history.sort_values(["rider_id", "Year", "race_order"], inplace=True)

    # initialise engineered columns
    for col in [
        "best_tdf_result",
        "best_other_gt_result",
        "best_recent_tdf_result",
        "best_recent_other_gt_result",
        "tdf_debut",
        "gt_debut",
        "rode_giro",
    ]:
        gt_history[col] = np.nan

    processed_rows: list[pd.DataFrame] = []

    for _, rider_df in gt_history.groupby("rider_id"):
        rider_df = rider_df.copy()
        best_tdf: float | int | np.nan = np.nan
        best_other: float | int | np.nan = np.nan
        seen_gt = False
        seen_tdf = False

        for idx, row in rider_df.iterrows():
            year, race, pos = row["Year"], row["Race"], row["Pos_clean"]

            # first‑ever GT / TdF flags
            if not seen_gt:
                rider_df.at[idx, "gt_debut"] = 1
                seen_gt = True
            if race == "Tour de France" and not seen_tdf:
                rider_df.at[idx, "tdf_debut"] = 1
                seen_tdf = True

            # career bests so far
            rider_df.at[idx, "best_tdf_result"] = best_tdf
            rider_df.at[idx, "best_other_gt_result"] = best_other

            # recent three‑year window bests
            recent = rider_df[(rider_df["Year"] < year) & (rider_df["Year"] >= year - 3)]
            rider_df.at[idx, "best_recent_tdf_result"] = recent.loc[
                recent["Race"] == "Tour de France", "Pos_clean"
            ].min()
            rider_df.at[idx, "best_recent_other_gt_result"] = recent.loc[
                recent["Race"].isin(["Giro d'Italia", "Vuelta a España"]), "Pos_clean"
            ].min()

            # rode Giro earlier in season?
            if race == "Tour de France":
                if year == 2020:
                    rider_df.at[idx, "rode_giro"] = 0  # chronological anomaly
                else:
                    rode_giro = rider_df[
                        (rider_df["Year"] == year)
                        & (rider_df["Race"] == "Giro d'Italia")
                        & (rider_df["race_order"] < row["race_order"])
                    ]
                    rider_df.at[idx, "rode_giro"] = 0 if rode_giro.empty else 1

            # update all‑time bests *after* current race
            if not np.isnan(pos):
                if race == "Tour de France":
                    best_tdf = pos if np.isnan(best_tdf) else min(best_tdf, pos)
                else:
                    best_other = pos if np.isnan(best_other) else min(best_other, pos)

        processed_rows.append(rider_df)

    best_gt = (
        pd.concat(processed_rows)
        .query("Race == 'Tour de France'")
        .sort_values(["rider_id", "Year", "race_order"])
        .reset_index(drop=True)
    )

    return best_gt[
        [
            "rider_id",
            "Year",
            "best_tdf_result",
            "best_other_gt_result",
            "best_recent_tdf_result",
            "best_recent_other_gt_result",
            "tdf_debut",
            "gt_debut",
            "rode_giro",
        ]
    ].rename(columns={"rider_id": "Rider_ID"})

# ---------------------------------------------------------------------
# Race‑results / ranking feature engineering
# ---------------------------------------------------------------------

def _prepare_race_tables(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create per‑rider, per‑year performance summary (incl. Year‑Before stats)."""

    # combine WorldTour & ProTour stage‑race results
    uwt_res, pt_res = data["uwt_res"].copy(), data["pt_res"].copy()
    uwt_res["CAT"], pt_res["CAT"] = "2.UWT", "2.Pro"
    res = pd.concat([uwt_res, pt_res], ignore_index=True)

    res["Year"] = res["YEAR"].astype(int)
    res["Rider_ID"] = res["Rider URL"].str.extract(r"r=(\d+)").astype(int)
    res["Race_ID"] = res["RACE_URL"].str.extract(r"r=(\d+)").astype(int)
    res = res[["Year", "Race_ID", "CAT", "RACE", "Rider_ID", "Born", "Pos"]]

    # ---------- race metadata (calendar position) ----------
    races = pd.concat([data["uwt_races"], data["pt_races"]])
    races = races[races["CAT"].isin(["2.UWT", "2.HC", "2.Pro"])].copy()
    races["Race_ID"] = races["RACE_URL"].str.extract(r"r=(\d+)").astype(int)
    races.rename(columns={"YEAR": "Year", "DATE": "Date"}, inplace=True)
    races = races[["Year", "Date", "Race_ID"]]

    # parse dd.mm‑dd.mm format -> start date
    races["Start_Date"] = pd.to_datetime(
        races["Date"].str.split("-").str[0] + "." + races["Year"].astype(str),
        format="%d.%m.%Y",
    )
    tdf_dates = races.loc[races["Race_ID"] == 17, ["Year", "Start_Date"]].rename(
        columns={"Start_Date": "Tour_Date"}
    )
    races = races.merge(tdf_dates, on="Year", how="left")
    races["Before_Tour"] = (races["Start_Date"] < races["Tour_Date"]).astype(int)
    races.drop(columns=["Date", "Start_Date", "Tour_Date"], inplace=True)

    # ---------- FirstCycling rankings ----------
    fc_rank = data["fc_rank"].copy()
    fc_rank["Year"] = fc_rank["Year"].astype(int)
    fc_rank["Rider_ID"] = fc_rank["rider_id"].astype(int)
    fc_rank = fc_rank.drop(columns=["Rider"]).drop_duplicates()

    # ---------- merge results with metadata & rankings ----------
    res_races = res.merge(races, on=["Year", "Race_ID"], how="left")
    res_races_fc = res_races.merge(
        fc_rank, on=["Rider_ID", "Year"], how="left", validate="m:1"
    ).rename(columns={"Points": "FC_Points"})

    # clean Pos -> int placeholders
    res_races_fc["Pos"] = (
        res_races_fc["Pos"]
        .replace({"DNF": 999, "DNS": 998, "DSQ": 997})
        .fillna(1000)
        .astype(int)
    )

    res_races_fc["TDF_Pos"] = np.where(res_races_fc["Race_ID"] == 17, res_races_fc["Pos"], 1000)
    res_races_fc["FC_Points"] = res_races_fc["FC_Points"].fillna(0)
    res_races_fc["FC_Pos"] = (
        res_races_fc.groupby("Year")["FC_Points"].rank(ascending=False, method="dense").astype(int)
    )

    # convenience helper to mark best positions --------------------------------
    def _best(col: str, before: bool, cat: str):
        mask = (res_races_fc["Before_Tour"] == (1 if before else 0)) & (
            res_races_fc["CAT"] == cat
        )
        if not before:
            mask &= res_races_fc["Race_ID"] != 17
        res_races_fc[col] = np.where(mask, res_races_fc["Pos"], np.nan)

    _best("Best_Pos_BT_UWT", before=True, cat="2.UWT")
    _best("Best_Pos_BT_PT", before=True, cat="2.Pro")
    _best("Best_Pos_AT_UWT", before=False, cat="2.UWT")
    _best("Best_Pos_AT_PT", before=False, cat="2.Pro")

    # aggregate to rider‑year level -------------------------------------------
    agg = res_races_fc.groupby(["Rider_ID", "Year", "Born"], as_index=False).agg(
        FC_Points=("FC_Points", "max"),
        FC_Pos=("FC_Pos", "max"),
        Best_Pos_BT_UWT=("Best_Pos_BT_UWT", "min"),
        Best_Pos_BT_PT=("Best_Pos_BT_PT", "min"),
        Best_Pos_AT_UWT=("Best_Pos_AT_UWT", "min"),
        Best_Pos_AT_PT=("Best_Pos_AT_PT", "min"),
        TDF_Pos=("TDF_Pos", "min"),
    )

    agg["Best_Pos_UWT"] = agg[["Best_Pos_BT_UWT", "Best_Pos_AT_UWT"]].min(axis=1)
    agg["Best_Pos_PT"] = agg[["Best_Pos_BT_PT", "Best_Pos_AT_PT"]].min(axis=1)

    # Year‑Before table --------------------------------------------------------
    yb = agg.drop(columns=["FC_Pos"]).copy()
    yb["Year"] += 1  # shift forward
    yb = yb.rename(
        columns={
            "FC_Points": "FC_Points_YB",
            "Best_Pos_BT_UWT": "Best_Pos_BT_UWT_YB",
            "Best_Pos_BT_PT": "Best_Pos_BT_PT_YB",
            "Best_Pos_AT_UWT": "Best_Pos_AT_UWT_YB",
            "Best_Pos_AT_PT": "Best_Pos_AT_PT_YB",
            "Best_Pos_UWT": "Best_Pos_UWT_YB",
            "Best_Pos_PT": "Best_Pos_PT_YB",
        }
    )

    full = agg.merge(
        yb[
            [
                "Rider_ID",
                "Year",
                "FC_Points_YB",
                "Best_Pos_BT_UWT_YB",
                "Best_Pos_BT_PT_YB",
                "Best_Pos_AT_UWT_YB",
                "Best_Pos_AT_PT_YB",
                "Best_Pos_UWT_YB",
                "Best_Pos_PT_YB",
            ]
        ],
        on=["Rider_ID", "Year"],
        how="left",
    )

    full["FC_Points_YB"] = full["FC_Points_YB"].fillna(0)
    full["FC_Pos_YB"] = (
        full.groupby("Year")["FC_Points_YB"].rank(ascending=False, method="dense").astype(int)
    )

    return full

# ---------------------------------------------------------------------
# Top‑level pipeline function
# ---------------------------------------------------------------------

def preprocess_tdf_data(folder_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Run the complete preprocessing pipeline.

    Parameters
    ----------
    folder_path : str
        Directory containing all raw CSVs.
    output_path : str | None, default=None
        If provided, the resulting CSV will be written to this path. If the
        path is relative, it is considered relative to *folder_path*.

    Returns
    -------
    pandas.DataFrame
        The final feature table ready for model training.
    """

    folder_path = str(folder_path)
    data = _load_raw(folder_path)

    race_table = _prepare_race_tables(data)
    gt_table = _prepare_gt_history(data["gt_history"])

    df = race_table.merge(gt_table, on=["Rider_ID", "Year"], how="left")

    # final tidy‑up -----------------------------------------------------------
    df["Age"] = df["Year"] - df["Born"]
    df.drop(columns=["Born"], inplace=True)

    # convert placeholder ints back to categorical strings where needed
    def _replace(series: pd.Series) -> pd.Series:
        return series.replace({1000: np.nan, 999: "DNF", 998: "DNS", 997: "DSQ"})

    df["TDF_Pos"] = _replace(df["TDF_Pos"])
    for col in [
        "Best_Pos_BT_UWT",
        "Best_Pos_AT_UWT_YB",
        "Best_Pos_UWT_YB",
        "Best_Pos_BT_PT",
        "Best_Pos_AT_PT_YB",
        "Best_Pos_PT_YB",
    ]:
        df[col] = df[col].replace({999: "DNF"})

    # reorder columns for readability ----------------------------------------
    col_order = [
        "Rider_ID",
        "Year",
        "Age",
        "TDF_Pos",
        "Best_Pos_BT_UWT",
        "Best_Pos_BT_PT",
        "Best_Pos_AT_UWT_YB",
        "Best_Pos_AT_PT_YB",
        "Best_Pos_UWT_YB",
        "Best_Pos_PT_YB",
        "FC_Points_YB",
        "FC_Pos_YB",
        "best_tdf_result",
        "best_other_gt_result",
        "best_recent_tdf_result",
        "best_recent_other_gt_result",
        "tdf_debut",
        "gt_debut",
        "rode_giro",
    ]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[col_order + remaining]

    # write CSV if requested --------------------------------------------------
    if output_path:
        out_path = Path(output_path)
        if not out_path.is_absolute():
            out_path = Path(folder_path) / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote prepared data to {out_path}")

    return df

# ---------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TdF datasets into feature table")
    parser.add_argument("folder", help="Folder containing raw CSV files")
    parser.add_argument("-o", "--output", help="Optional output CSV filename", default=None)
    args = parser.parse_args()

    preprocess_tdf_data(args.folder, args.output)
