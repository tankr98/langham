"""
langham_model.py
================
Langham Pasadena — Displacement Value Regression Model
Backend module for UI integration.

SALES REP INPUTS (per night):
  - date
  - group_rooms   (rooms needed that night)
  - is_holiday    (boolean flag)

AUTOMATICALLY RESOLVED (no rep input):
  - non_group_rooms  looked up from historical median by month + day-of-week
                     (occupancy_pct is not used — removing it preserves R² via
                      non_group_rooms which is hotel inventory data, not gameable)

DISPLACEMENT VALUE FORMULA
--------------------------
Per night:
    Adj Rev per Room  = (Predicted ADR − 150) + (Predicted F&B per Night × 0.20)
    Night Contribution = Predicted LCR × Group Rooms × Adj Rev per Room

Total booking:
    Total Displacement  = Σ Night Contributions
    Parking Cost        = Excess Parking Days × $8,000
    Displacement Value  = Total Displacement − Group Profit − Parking Cost

Negative = group profit exceeds displaced leisure → Book the group
Positive = leisure revenue exceeds group profit   → Decline the group
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHEET_NAME   = "Historical Data Master"
HEADER_ROW   = 8
RANDOM_STATE = 42
TEST_SIZE    = 0.20
PARKING_COST_PER_DAY = 8_000

DOW_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

# Features used by all three models
# NOTE: occupancy_pct intentionally excluded — non_group_rooms from historical
# lookup carries equivalent signal without rep-gaming risk
FEATURES = [
    "month",           # Month as integer 1–12
    "day",             # Day of month 1–31
    "dow_num",         # Day of week 0=Monday … 6=Sunday
    "is_holiday",      # 1 if hotel holiday period
    "group_rooms",     # Group rooms booked that night (rep input)
    "non_group_rooms", # Auto-resolved from historical median lookup
]

FILTERS = {
    "lcr":         (0.0,   1.0),
    "leisure_adr": (100.0, 700.0),
    "leisure_fnb": (0.0,   300.0),
}


# ---------------------------------------------------------------------------
# DisplacementModel
# ---------------------------------------------------------------------------

class DisplacementModel:
    """
    Trains three regression models on historical Langham data.
    non_group_rooms is resolved automatically from a historical median lookup
    (month × day-of-week) so sales reps never need to enter it.
    """

    def __init__(self):
        self.models         = {}
        self.scalers        = {}
        self.metrics        = {}
        self.ngr_lookup     = {}   # (month, dow_num) -> median non_group_rooms
        self.ngr_fallback   = 161  # overall median, used if lookup misses
        self.is_trained     = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, excel_path: str, verbose: bool = True) -> dict:
        """
        Load data, build non_group_rooms lookup, train all three models.

        Parameters
        ----------
        excel_path : path to Langham_True_Profit_Model_vLocal.xlsx
        verbose    : print progress and summary

        Returns
        -------
        dict of performance metrics per target
        """
        raw = pd.read_excel(excel_path, sheet_name=SHEET_NAME, header=HEADER_ROW)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")

        # Build non_group_rooms historical lookup BEFORE filtering
        # (use full dataset so lookup covers all month/dow combos)
        self._build_ngr_lookup(raw)

        df = self._clean(raw)

        if verbose:
            print(f"Clean training rows: {len(df)}")
            print(f"Non-group rooms lookup: {len(self.ngr_lookup)} month×dow combinations")

        targets = {
            "lcr": "lcr",
            "adr": "leisure_adr",
            "fnb": "leisure_fnb",
        }

        for key, col in targets.items():
            if verbose:
                print(f"\n--- Training {key.upper()} model ---")
            model, scaler, metrics = self._train_one(df, col, verbose=verbose)
            self.models[key]  = model
            self.scalers[key] = scaler
            self.metrics[key] = metrics

        self.is_trained = True

        if verbose:
            self._print_summary()

        return self.metrics

    def _build_ngr_lookup(self, raw: pd.DataFrame):
        """Build median non_group_rooms by (month, dow_num) from full raw data."""
        df = raw.copy()
        df["dow_num"] = df["Day of Week"].map(DOW_MAP)
        df = df.dropna(subset=["Month", "dow_num", "Non-Group Rooms"])
        df = df[df["Non-Group Rooms"] > 0]

        lookup = (
            df.groupby(["Month", "dow_num"])["Non-Group Rooms"]
            .median()
        )
        self.ngr_lookup   = {(int(m), int(d)): float(v) for (m, d), v in lookup.items()}
        self.ngr_fallback = float(df["Non-Group Rooms"].median())

    def _resolve_non_group_rooms(self, month: int, dow_num: int) -> float:
        """Look up historical median non_group_rooms for a given month + day-of-week."""
        return self.ngr_lookup.get((month, dow_num), self.ngr_fallback)

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rename columns, apply artifact filters, encode features."""
        df = raw.rename(columns={
            "Month":                                "month",
            "Day":                                  "day",
            "Day of Week":                          "day_of_week_name",
            "Holiday":                              "is_holiday",
            "Group Rooms":                          "group_rooms",
            "Non-Group Rooms":                      "non_group_rooms",
            "Leisure Capture Rate":                 "lcr",
            "Leisure Room Revenue per Night (ADR)": "leisure_adr",
            "Leisure F&B Revenue per Night":        "leisure_fnb",
        })

        df = df.dropna(subset=[
            "Date", "month", "day", "group_rooms", "non_group_rooms",
            "lcr", "leisure_adr", "leisure_fnb",
        ])

        df["dow_num"]    = df["day_of_week_name"].map(DOW_MAP)
        df["is_holiday"] = df["is_holiday"].astype(int)

        df = df[df["non_group_rooms"] > 0]
        df = df[
            (df["lcr"]         >= FILTERS["lcr"][0])         & (df["lcr"]         <= FILTERS["lcr"][1])         &
            (df["leisure_adr"] >= FILTERS["leisure_adr"][0]) & (df["leisure_adr"] <= FILTERS["leisure_adr"][1]) &
            (df["leisure_fnb"] >= FILTERS["leisure_fnb"][0]) & (df["leisure_fnb"] <= FILTERS["leisure_fnb"][1])
        ]

        return df

    def _train_one(self, df: pd.DataFrame, target_col: str, verbose: bool = True):
        X = df[FEATURES].values
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        candidates = {
            "Ridge Regression":  Ridge(alpha=10),
            "Random Forest":     RandomForestRegressor(
                                     n_estimators=300, max_depth=8,
                                     random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(
                                     n_estimators=300, learning_rate=0.05,
                                     max_depth=4, subsample=0.8,
                                     random_state=RANDOM_STATE),
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        results = []
        for name, model in candidates.items():
            scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="r2")
            results.append({"name": name, "model": model,
                            "cv_mean": scores.mean(), "cv_std": scores.std()})
            if verbose:
                print(f"  {name:<22} CV R2: {scores.mean():.4f} ± {scores.std():.4f}")

        best       = max(results, key=lambda r: r["cv_mean"])
        best_model = best["model"]
        best_model.fit(X_train_sc, y_train)
        y_pred = best_model.predict(X_test_sc)

        metrics = {
            "winner": best["name"],
            "r2":     round(r2_score(y_test, y_pred), 4),
            "mae":    round(mean_absolute_error(y_test, y_pred), 3),
            "rmse":   round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        }
        if verbose:
            print(f"  Winner: {metrics['winner']}  |  "
                  f"MAE: {metrics['mae']}  RMSE: {metrics['rmse']}  R2: {metrics['r2']}")

        return best_model, scaler, metrics

    def _print_summary(self):
        labels = {
            "lcr": "Leisure Capture Rate",
            "adr": "Leisure ADR ($/room night)",
            "fnb": "Leisure F&B ($/room night)",
        }
        confidence = lambda r2: (
            "High" if r2 > 0.85 else "Good" if r2 > 0.70 else
            "Moderate" if r2 > 0.50 else "Directional"
        )
        print("\n" + "=" * 65)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 65)
        print(f"{'Target':<30} {'R2':>6} {'MAE':>8} {'RMSE':>8}  Confidence")
        for key, label in labels.items():
            m = self.metrics[key]
            print(f"{label:<30} {m['r2']:>6}  {m['mae']:>7}  {m['rmse']:>7}  {confidence(m['r2'])}")
        print("\nNote: non_group_rooms resolved automatically from historical")
        print("      median lookup (month x day-of-week). No rep input needed.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_night(self, date: str, group_rooms: float,
                      is_holiday: bool = False) -> dict:
        """
        Predict LCR, Leisure ADR, and Leisure F&B per Night for a single night.
        non_group_rooms is resolved automatically from historical lookup.

        Parameters
        ----------
        date        : "YYYY-MM-DD"
        group_rooms : number of group rooms booked that night (rep input)
        is_holiday  : True if hotel holiday period

        Returns
        -------
        dict with all predicted values and the night's displacement contribution
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        ts      = pd.Timestamp(date)
        month   = ts.month
        day     = ts.day
        dow_num = ts.dayofweek   # 0=Monday … 6=Sunday

        # Resolve non_group_rooms from historical lookup — no rep input
        non_group_rooms = self._resolve_non_group_rooms(month, dow_num)

        row_vec = np.array([[
            month, day, dow_num, int(is_holiday),
            group_rooms, non_group_rooms,
        ]])

        def _pred(key):
            return float(
                self.models[key].predict(
                    self.scalers[key].transform(row_vec)
                )[0]
            )

        pred_lcr = float(np.clip(_pred("lcr"), 0.0, 1.0))
        pred_adr = float(max(0.0, _pred("adr")))
        pred_fnb = float(max(0.0, _pred("fnb")))

        # Displacement contribution for this night:
        # Adj Rev per Room  = (ADR − 150) + (F&B × 0.20)
        # Night Contribution = LCR × Group Rooms × Adj Rev per Room
        adj_rev_per_room   = (pred_adr - 150.0) + (pred_fnb * 0.20)
        night_contribution = pred_lcr * group_rooms * adj_rev_per_room

        return {
            "date":                  date,
            "day_of_week":           ts.day_name(),
            "is_holiday":            is_holiday,
            "group_rooms":           group_rooms,
            "non_group_rooms_used":  round(non_group_rooms, 1),
            "predicted_lcr":         round(pred_lcr, 4),
            "predicted_adr":         round(pred_adr, 2),
            "predicted_fnb":         round(pred_fnb, 2),
            "adj_rev_per_room":      round(adj_rev_per_room, 2),
            "night_contribution":    round(night_contribution, 2),
        }

    def predict_booking(self, nights: list[dict], group_profit: float,
                        excess_parking_days: int = 0) -> dict:
        """
        Evaluate a multi-night group booking.

        Parameters
        ----------
        nights : list of dicts, each containing:
            - date        : "YYYY-MM-DD"
            - group_rooms : float
            - is_holiday  : bool (optional, default False)

        group_profit : float
            Total group profit for the booking ($). Entered by sales rep.
            Formula: (Group ADR − 150) × total room nights
                     + (0.20 × F&B revenue)
                     + Additional Revenue − Costs

        excess_parking_days : int
            Number of days requiring excess parking.
            Cost = excess_parking_days × $8,000, deducted from group_profit.

        Returns
        -------
        dict:
            nights               -> per-night prediction dicts
            total_displacement   -> Σ night_contribution across all nights
            parking_cost         -> excess_parking_days × $8,000
            adjusted_group_profit-> group_profit − parking_cost
            displacement_value   -> total_displacement − adjusted_group_profit
            verdict              -> "Book the group" or "Decline the group"
            model_metrics        -> R2/MAE/RMSE for each model
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        night_results = []
        for n in nights:
            night_results.append(self.predict_night(
                date        = n["date"],
                group_rooms = n["group_rooms"],
                is_holiday  = n.get("is_holiday", False),
            ))

        total_displacement    = sum(r["night_contribution"] for r in night_results)
        parking_cost          = excess_parking_days * PARKING_COST_PER_DAY
        adjusted_group_profit = group_profit - parking_cost
        displacement_value    = total_displacement - adjusted_group_profit
        verdict = "Book the group" if displacement_value < 0 else "Decline the group"

        return {
            "nights":                night_results,
            "total_displacement":    round(total_displacement, 2),
            "parking_cost":          round(parking_cost, 2),
            "adjusted_group_profit": round(adjusted_group_profit, 2),
            "displacement_value":    round(displacement_value, 2),
            "verdict":               verdict,
            "model_metrics":         self.metrics,
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
        "Langham_True_Profit_Model_vLocal.xlsx"

    print("Training model...")
    model = DisplacementModel()
    model.train(EXCEL_PATH, verbose=True)

    print("\n" + "=" * 65)
    print("SAMPLE BOOKING: May 19–21 2026, varying group rooms, 1 parking day")
    print("=" * 65)

    result = model.predict_booking(
        nights=[
            {"date": "2026-05-19", "group_rooms": 10,  "is_holiday": False},
            {"date": "2026-05-20", "group_rooms": 15,  "is_holiday": False},
            {"date": "2026-05-21", "group_rooms": 10,  "is_holiday": False},
        ],
        group_profit=18000.0,
        excess_parking_days=1,
    )

    print("\nPer-night breakdown:")
    for n in result["nights"]:
        print(f"  {n['date']} ({n['day_of_week'][:3]})  "
              f"NGR={n['non_group_rooms_used']:.0f}  "
              f"LCR={n['predicted_lcr']:.3f}  "
              f"ADR=${n['predicted_adr']:,.0f}  "
              f"F&B=${n['predicted_fnb']:,.0f}  "
              f"AdjRev=${n['adj_rev_per_room']:,.0f}  "
              f"Contrib=${n['night_contribution']:,.0f}")

    print(f"\nTotal Displacement:     ${result['total_displacement']:,.2f}")
    print(f"Parking Cost:           ${result['parking_cost']:,.2f}")
    print(f"Adj. Group Profit:      ${result['adjusted_group_profit']:,.2f}")
    print(f"Displacement Value:     ${result['displacement_value']:,.2f}")
    print(f"Verdict:                {result['verdict']}")
