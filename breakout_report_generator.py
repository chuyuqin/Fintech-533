from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shinybroker as sb


OUTPUT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = OUTPUT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
DATA_DIR = DOCS_DIR / "data"

UNIVERSE = ["NVDA", "MSFT", "META", "AMZN", "AAPL", "MU", "AVGO", "LLY"]
DATA_DURATION = "2 Y"
HOURLY_DATA_DURATION = "1 Y"
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
SLIPPAGE_BPS = 2
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
RISK_FREE_RATE = 0.02

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "9999"))

WALKFORWARD_TRAIN_DAYS = 252
WALKFORWARD_TEST_DAYS = 63
WALKFORWARD_STEP_DAYS = 63

BREAKOUT_LOOKBACK_DAYS = 20
TREND_WINDOW_DAYS = 50
TREND_METHOD = "EMA"
ATR_WINDOW = 14
STOP_ATR_MULTIPLE = 1.2
TARGET_ATR_MULTIPLE = 2.4
MAX_HOLD_DAYS = 5
MIN_VOLUME_RATIO = 0.80

PARAM_GRID = [
    {
        "breakout_lookback": BREAKOUT_LOOKBACK_DAYS,
        "trend_window": TREND_WINDOW_DAYS,
        "trend_method": TREND_METHOD,
        "atr_window": ATR_WINDOW,
        "stop_atr": STOP_ATR_MULTIPLE,
        "target_atr": TARGET_ATR_MULTIPLE,
        "max_hold_days": MAX_HOLD_DAYS,
        "min_volume_ratio": MIN_VOLUME_RATIO,
    }
]

QUALITATIVE_OVERLAY = {
    "NVDA": {
        "score": 4.0,
        "reason": "Exceptional liquidity and strong secular AI narrative, but gap risk is high around crowded positioning.",
    },
    "MSFT": {
        "score": 4.8,
        "reason": "Deep liquidity, strong institutional participation, and smoother trend behavior than many high-beta names.",
    },
    "META": {
        "score": 4.1,
        "reason": "Very liquid with strong momentum phases, though headline sensitivity can create sharper reversals.",
    },
    "AMZN": {
        "score": 4.0,
        "reason": "High liquidity and persistent trend phases, but sometimes less clean on short-horizon breakout follow-through.",
    },
    "AAPL": {
        "score": 4.5,
        "reason": "Elite liquidity and broad participation, though the stock can spend long periods in mature ranges.",
    },
    "MU": {
        "score": 4.3,
        "reason": "Semiconductor cyclicality often creates tradable expansions, but earnings and guidance risk are meaningful.",
    },
    "AVGO": {
        "score": 4.2,
        "reason": "Strong trend persistence and liquidity, but occasional post-event gaps add idiosyncratic risk.",
    },
    "LLY": {
        "score": 3.7,
        "reason": "Excellent liquidity and durable trend leadership, though healthcare catalysts can distort breakout behavior.",
    },
}


@dataclass(frozen=True)
class WindowResult:
    window_summary: pd.DataFrame
    parameter_history: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass(frozen=True)
class SymbolData:
    daily: pd.DataFrame
    hourly: pd.DataFrame


def ensure_directories() -> None:
    for path in [DOCS_DIR, ASSETS_DIR, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def make_stock_contract(symbol: str) -> sb.Contract:
    return sb.Contract(
        {
            "symbol": symbol,
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    )


def standardize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).lower() for col in out.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"IBKR history for {symbol} is missing columns: {sorted(missing)}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["symbol"] = symbol
    out = (
        out.loc[:, ["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
        .dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out["date"] = out["timestamp"].dt.normalize()
    return out


def fetch_ibkr_price_data(
    symbol: str,
    duration: str = DATA_DURATION,
    hourly_duration: str = HOURLY_DATA_DURATION,
    host: str = IB_HOST,
    port: int = IB_PORT,
    client_id: int = IB_CLIENT_ID,
    timeout: int = 10,
) -> SymbolData:
    try:
        asset = make_stock_contract(symbol)
        hourly_raw = sb.fetch_historical_data(
            asset,
            endDateTime="",
            durationStr=hourly_duration,
            barSizeSetting="1 hour",
            whatToShow="Trades",
            useRTH=True,
            host=host,
            port=port,
            client_id=client_id,
            timeout=timeout,
        )
        daily_raw = sb.fetch_historical_data(
            asset,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="Trades",
            useRTH=True,
            host=host,
            port=port,
            client_id=client_id,
            timeout=timeout,
        )
    except Exception as exc:
        raise RuntimeError(
            f"IBKR fetch failed for {symbol}. Confirm that TWS or IB Gateway is running, "
            f"API access is enabled, and the script settings match host={host}, port={port}, "
            f"client_id={client_id}. If the port is correct, try a different client_id because "
            f"the current one may already be in use."
        ) from exc
    if not isinstance(hourly_raw, dict) or "hst_dta" not in hourly_raw:
        raise ValueError(f"Malformed IBKR hourly response for {symbol}: {hourly_raw}")
    if not isinstance(daily_raw, dict) or "hst_dta" not in daily_raw:
        raise ValueError(f"Malformed IBKR daily response for {symbol}: {daily_raw}")

    hourly_df = pd.DataFrame(hourly_raw["hst_dta"])
    daily_df = pd.DataFrame(daily_raw["hst_dta"])
    if hourly_df.empty or daily_df.empty:
        raise ValueError(f"Empty IBKR history for {symbol}")

    return SymbolData(
        daily=standardize_ohlcv(daily_df, symbol),
        hourly=standardize_ohlcv(hourly_df, symbol),
    )


def build_daily_features(daily_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = daily_df.copy().reset_index(drop=True)
    out["prev_close"] = out["close"].shift(1)
    tr_components = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["prev_close"]).abs(),
            (out["low"] - out["prev_close"]).abs(),
        ],
        axis=1,
    )
    out["true_range"] = tr_components.max(axis=1)
    out["atr"] = out["true_range"].rolling(params["atr_window"]).mean()
    if params["trend_method"] == "EMA":
        out["trend_line"] = out["close"].ewm(span=params["trend_window"], adjust=False).mean()
    else:
        out["trend_line"] = out["close"].rolling(params["trend_window"]).mean()

    out["avg_volume"] = out["volume"].rolling(20).mean()
    out["breakout_level"] = out["high"].shift(1).rolling(params["breakout_lookback"]).max()
    out["breakout_ready"] = (
        (out["close"].shift(1) > out["trend_line"].shift(1))
        & (out["volume"].shift(1) >= params["min_volume_ratio"] * out["avg_volume"].shift(1))
    )
    return out


def identify_breakout(
    row: pd.Series,
    breakout_level_col: str = "breakout_level",
    price_col: str = "close",
    prior_price_col: str = "prior_hour_close",
    readiness_col: str = "breakout_ready",
) -> bool:
    breakout_level = row.get(breakout_level_col, np.nan)
    current_price = row.get(price_col, np.nan)
    prior_price = row.get(prior_price_col, np.nan)
    ready_flag = bool(row.get(readiness_col, False))

    if not ready_flag:
        return False
    if not np.isfinite(breakout_level) or not np.isfinite(current_price) or not np.isfinite(prior_price):
        return False
    return bool(prior_price <= breakout_level and current_price > breakout_level)


def build_market_data(symbol_data: SymbolData, params: dict) -> pd.DataFrame:
    daily_features = build_daily_features(symbol_data.daily, params)
    feature_cols = ["date", "breakout_level", "atr", "trend_line", "avg_volume", "breakout_ready"]
    next_day_features = daily_features.loc[:, feature_cols].copy()
    next_day_features["date"] = next_day_features["date"] + pd.Timedelta(days=1)

    market = symbol_data.hourly.merge(next_day_features, on="date", how="left")
    market["prior_hour_close"] = market["close"].shift(1)
    market["signal_bar"] = market.apply(identify_breakout, axis=1)
    return market


def estimate_transaction_cost(shares: int, price: float) -> float:
    notional = abs(shares * price)
    slippage_cost = notional * (SLIPPAGE_BPS / 10_000)
    commission_cost = max(MIN_COMMISSION, abs(shares) * COMMISSION_PER_SHARE)
    return float(slippage_cost + commission_cost)


def classify_trade_outcome(exit_reason: str, net_pnl: float) -> str:
    if exit_reason == "stop_loss":
        return "Stop-loss triggered"
    if exit_reason == "profit_target":
        return "Successful"
    if exit_reason == "timeout":
        return "Timed out"
    return "Successful" if net_pnl > 0 else "Timed out"


def run_hourly_breakout_backtest(
    market_data: pd.DataFrame,
    params: dict,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    equity = initial_capital
    shares = 0
    active_trade: dict | None = None
    trades: list[dict] = []
    equity_curve: list[dict] = []

    for idx, row in market_data.iterrows():
        date = row["timestamp"]
        action = "HOLD"

        if active_trade is not None:
            holding_days = (date - active_trade["entry_date"]).total_seconds() / 86_400
            hit_stop = row["low"] <= active_trade["stop_price"]
            hit_target = row["high"] >= active_trade["target_price"]
            timed_out = holding_days >= params["max_hold_days"]
            last_bar = idx == len(market_data) - 1

            if hit_stop or hit_target or timed_out or last_bar:
                if hit_stop:
                    exit_price = active_trade["stop_price"]
                    exit_reason = "stop_loss"
                elif hit_target:
                    exit_price = active_trade["target_price"]
                    exit_reason = "profit_target"
                else:
                    exit_price = float(row["close"])
                    exit_reason = "timeout" if timed_out else "end_of_data"

                exit_cost = estimate_transaction_cost(shares, exit_price)
                gross_pnl = shares * (exit_price - active_trade["entry_price"])
                net_pnl = gross_pnl - active_trade["entry_cost"] - exit_cost
                starting_trade_value = active_trade["entry_price"] * shares
                equity += net_pnl

                trades.append(
                    {
                        "symbol": active_trade["symbol"],
                        "entry_date": active_trade["entry_date"],
                        "exit_date": date,
                        "entry_price": active_trade["entry_price"],
                        "exit_price": exit_price,
                        "shares": shares,
                        "direction": "long",
                        "breakout_level": active_trade["breakout_level"],
                        "atr_at_entry": active_trade["atr_at_entry"],
                        "stop_price": active_trade["stop_price"],
                        "target_price": active_trade["target_price"],
                        "bars_held": round(holding_days * 6.5, 2),
                        "gross_pnl": gross_pnl,
                        "transaction_costs": active_trade["entry_cost"] + exit_cost,
                        "net_pnl": net_pnl,
                        "return_on_trade": net_pnl / starting_trade_value if starting_trade_value else np.nan,
                        "exit_reason": exit_reason,
                        "trade_outcome": classify_trade_outcome(exit_reason, net_pnl),
                    }
                )
                shares = 0
                active_trade = None
                action = "EXIT"

        if (
            active_trade is None
            and bool(row["signal_bar"])
            and np.isfinite(row["atr"])
            and row["atr"] > 0
            and np.isfinite(row["breakout_level"])
        ):
            entry_price = float(row["close"])
            atr_value = float(row["atr"])
            stop_distance = params["stop_atr"] * atr_value
            # Position size is chosen so the initial stop risks about 1% of equity.
            risk_budget = equity * RISK_PER_TRADE
            shares = max(1, int(risk_budget / max(stop_distance, 1e-6)))
            entry_cost = estimate_transaction_cost(shares, entry_price)
            equity -= entry_cost
            action = "ENTER_LONG"
            active_trade = {
                "symbol": row["symbol"],
                "entry_date": date,
                "entry_price": entry_price,
                "entry_cost": entry_cost,
                "breakout_level": float(row["breakout_level"]),
                "atr_at_entry": atr_value,
                "stop_price": entry_price - params["stop_atr"] * atr_value,
                "target_price": entry_price + params["target_atr"] * atr_value,
            }

        mark_to_market_equity = equity
        if active_trade is not None:
            mark_to_market_equity += shares * (float(row["close"]) - active_trade["entry_price"])

        equity_curve.append(
            {
                "timestamp": date,
                "symbol": row["symbol"],
                "equity": mark_to_market_equity,
                "action": action,
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_curve), float(equity)


def summarize_performance(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    base = {
        "ending_equity": np.nan,
        "total_return": np.nan,
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "sharpe_ratio": np.nan,
        "max_drawdown": np.nan,
        "num_trades": int(len(trades)),
        "avg_return_per_trade": np.nan,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "expectancy_dollars": np.nan,
    }
    if equity_curve.empty:
        return base

    ec = equity_curve.copy().sort_values("timestamp").reset_index(drop=True)
    ec["bar_return"] = ec["equity"].pct_change().fillna(0.0)
    total_return = ec["equity"].iloc[-1] / initial_capital - 1
    hourly_periods_per_year = 252 * 6.5
    annualized_return = (1 + total_return) ** (hourly_periods_per_year / max(len(ec), 1)) - 1 if len(ec) > 1 else np.nan
    annualized_volatility = ec["bar_return"].std(ddof=1) * math.sqrt(hourly_periods_per_year) if len(ec) > 1 else np.nan
    excess_return = annualized_return - risk_free_rate if pd.notna(annualized_return) else np.nan
    sharpe_ratio = excess_return / annualized_volatility if pd.notna(annualized_volatility) and annualized_volatility > 0 else np.nan
    running_peak = ec["equity"].cummax()
    drawdown = ec["equity"] / running_peak - 1
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan

    summary = {
        **base,
        "ending_equity": float(ec["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return) if pd.notna(annualized_return) else np.nan,
        "annualized_volatility": float(annualized_volatility) if pd.notna(annualized_volatility) else np.nan,
        "sharpe_ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan,
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
    }

    if not trades.empty:
        gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
        gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
        summary["avg_return_per_trade"] = float(trades["return_on_trade"].mean())
        summary["win_rate"] = float((trades["net_pnl"] > 0).mean())
        summary["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan
        summary["expectancy_dollars"] = float(trades["net_pnl"].mean())

    return summary


def score_parameter_set(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    summary = summarize_performance(trades, equity_curve)
    sharpe = -999 if pd.isna(summary["sharpe_ratio"]) else summary["sharpe_ratio"]
    total_return = -999 if pd.isna(summary["total_return"]) else summary["total_return"]
    max_dd_penalty = 0 if pd.isna(summary["max_drawdown"]) else abs(summary["max_drawdown"])
    trade_bonus = min(summary["num_trades"], 10) / 100
    return sharpe + 0.5 * total_return - 0.25 * max_dd_penalty + trade_bonus


def run_walkforward_for_symbol(symbol_data: SymbolData, symbol: str) -> WindowResult:
    daily_df = symbol_data.daily.copy().sort_values("timestamp").reset_index(drop=True)
    if len(daily_df) < WALKFORWARD_TRAIN_DAYS + WALKFORWARD_TEST_DAYS:
        raise ValueError(f"Not enough daily history for walk-forward on {symbol}")

    window_rows: list[dict] = []
    chosen_rows: list[dict] = []
    all_test_trades: list[pd.DataFrame] = []
    all_test_equity: list[pd.DataFrame] = []

    start_test_idx = WALKFORWARD_TRAIN_DAYS
    oos_capital = INITIAL_CAPITAL

    while start_test_idx + WALKFORWARD_TEST_DAYS <= len(daily_df):
        train_days = daily_df.iloc[start_test_idx - WALKFORWARD_TRAIN_DAYS : start_test_idx].copy()
        test_days = daily_df.iloc[start_test_idx : start_test_idx + WALKFORWARD_TEST_DAYS].copy()

        train_start = train_days["date"].iloc[0]
        train_end = train_days["date"].iloc[-1]
        test_start = test_days["date"].iloc[0]
        test_end = test_days["date"].iloc[-1]

        hourly_test = symbol_data.hourly.loc[
            (symbol_data.hourly["date"] >= test_start) & (symbol_data.hourly["date"] <= test_end)
        ].copy()

        best_params = PARAM_GRID[0].copy()
        best_row = {
            "symbol": symbol,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "param_id": 0,
            "score": np.nan,
            **best_params,
        }

        test_market = build_market_data(SymbolData(daily=test_days, hourly=hourly_test), best_params)
        test_trades, test_equity, oos_capital = run_hourly_breakout_backtest(test_market, best_params, oos_capital)
        test_perf = summarize_performance(
            test_trades,
            test_equity,
            initial_capital=float(test_equity["equity"].iloc[0]) if not test_equity.empty else oos_capital,
        )

        window_rows.append(
            {
                "symbol": symbol,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "chosen_param_id": int(best_row["param_id"]),
                **best_params,
                **test_perf,
            }
        )
        chosen_rows.append(best_row)

        if not test_trades.empty:
            temp_trades = test_trades.copy()
            temp_trades["test_start"] = test_start
            temp_trades["test_end"] = test_end
            temp_trades["chosen_param_id"] = int(best_row["param_id"])
            all_test_trades.append(temp_trades)

        temp_equity = test_equity.copy()
        temp_equity["test_start"] = test_start
        temp_equity["test_end"] = test_end
        temp_equity["chosen_param_id"] = int(best_row["param_id"])
        all_test_equity.append(temp_equity)

        start_test_idx += WALKFORWARD_STEP_DAYS

    trade_df = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    equity_df = pd.concat(all_test_equity, ignore_index=True) if all_test_equity else pd.DataFrame()
    return WindowResult(
        window_summary=pd.DataFrame(window_rows),
        parameter_history=pd.DataFrame(chosen_rows),
        trades=trade_df,
        equity_curve=equity_df,
    )


def select_focus_asset(asset_summary: pd.DataFrame) -> str:
    ranked = asset_summary.copy()
    ranked["qualitative_score"] = ranked["symbol"].map(
        lambda symbol: QUALITATIVE_OVERLAY.get(symbol, {}).get("score", 0.0)
    )
    ranked["selection_score"] = (
        0.55 * ranked["avg_test_sharpe"].fillna(-999)
        + 1.75 * ranked["avg_test_return"].fillna(-999)
        - 0.50 * ranked["avg_max_drawdown"].abs().fillna(0)
        + np.minimum(ranked["trade_count"], 15) / 12
        + 0.35 * ranked["qualitative_score"]
    )
    ranked = ranked.sort_values(["selection_score", "trade_count"], ascending=False)
    return str(ranked.iloc[0]["symbol"])


def money(value: float) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "N/A"


def pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "N/A"


def num(value: float) -> str:
    return f"{value:.2f}" if pd.notna(value) else "N/A"


def save_plot_images(
    focus_symbol: str,
    focus_equity: pd.DataFrame,
    focus_trades: pd.DataFrame,
    outcome_counts: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(focus_equity["timestamp"], focus_equity["equity"], color="#0b6e4f", linewidth=2)
    ax.set_title(f"{focus_symbol} Out-of-Sample Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#0b6e4f", "#d94841", "#6c757d"]
    ax.bar(outcome_counts["trade_outcome"], outcome_counts["count"], color=colors[: len(outcome_counts)])
    ax.set_title(f"{focus_symbol} Trade Outcomes")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Trade count")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "trade_outcomes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(focus_trades["net_pnl"], bins=20, color="#1f77b4", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{focus_symbol} Net PnL per Trade")
    ax.set_xlabel("Net PnL ($)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "trade_pnl_histogram.png", dpi=160)
    plt.close(fig)


def make_metric_cards(metrics: Iterable[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, description in metrics:
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <p>{description}</p>
            </div>
            """
        )
    return "\n".join(cards)


def dataframe_to_html(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_html(index=index, classes="data-table", border=0)


def write_site(
    focus_symbol: str,
    focus_params: dict,
    focus_summary: dict,
    asset_summary: pd.DataFrame,
    focus_trades: pd.DataFrame,
    focus_windows: pd.DataFrame,
    outcome_counts: pd.DataFrame,
    asset_selection_paragraph: str,
) -> None:
    metrics_html = make_metric_cards(
        [
            (
                "Average return per trade",
                pct(focus_summary["avg_return_per_trade"]),
                "Mean net return across all out-of-sample trades. This shows whether the typical trade added value after costs.",
            ),
            (
                "Annualized Sharpe ratio",
                num(focus_summary["sharpe_ratio"]),
                f"Computed from daily out-of-sample equity returns using a {RISK_FREE_RATE:.0%} annual risk-free rate assumption.",
            ),
            (
                "Max drawdown",
                pct(focus_summary["max_drawdown"]),
                "Largest peak-to-trough equity decline during the out-of-sample test. Smaller magnitude means the ride was smoother.",
            ),
            (
                "Win rate",
                pct(focus_summary["win_rate"]),
                "Share of trades with positive net PnL. It complements expectancy because high win rates can still lose money if losses are too large.",
            ),
            (
                "Profit factor",
                num(focus_summary["profit_factor"]),
                "Gross profits divided by gross losses. Values above 1.0 indicate the winners outweighed the losers.",
            ),
            (
                "Expectancy",
                money(focus_summary["expectancy_dollars"]),
                "Average dollar PnL per trade after slippage and commissions.",
            ),
        ]
    )

    top_windows = focus_windows.loc[
        :, ["test_start", "test_end", "breakout_lookback", "trend_method", "stop_atr", "target_atr", "total_return", "sharpe_ratio", "num_trades"]
    ].copy()
    top_windows["test_start"] = pd.to_datetime(top_windows["test_start"]).dt.strftime("%Y-%m-%d")
    top_windows["test_end"] = pd.to_datetime(top_windows["test_end"]).dt.strftime("%Y-%m-%d")
    top_windows["total_return"] = top_windows["total_return"].map(pct)
    top_windows["sharpe_ratio"] = top_windows["sharpe_ratio"].map(num)

    blotter = focus_trades.loc[
        :,
        [
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "direction",
            "bars_held",
            "exit_reason",
            "trade_outcome",
            "net_pnl",
            "return_on_trade",
        ],
    ].copy()
    blotter["entry_date"] = pd.to_datetime(blotter["entry_date"]).dt.strftime("%Y-%m-%d")
    blotter["exit_date"] = pd.to_datetime(blotter["exit_date"]).dt.strftime("%Y-%m-%d")
    for col in ["entry_price", "exit_price", "net_pnl"]:
        blotter[col] = blotter[col].map(lambda x: f"{x:,.2f}")
    blotter["return_on_trade"] = blotter["return_on_trade"].map(lambda x: f"{x:.2%}")

    ranking_table = asset_summary.copy()
    for col in ["avg_test_return", "avg_test_sharpe", "avg_max_drawdown", "win_rate"]:
        if col in ranking_table.columns:
            if "return" in col or "drawdown" in col or col == "win_rate":
                ranking_table[col] = ranking_table[col].map(pct)
            else:
                ranking_table[col] = ranking_table[col].map(num)
    for col in ["total_return", "sharpe_ratio"]:
        if col in ranking_table.columns:
            if col == "total_return":
                ranking_table[col] = ranking_table[col].map(pct)
            else:
                ranking_table[col] = ranking_table[col].map(num)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Breakout Strategy Walk-Forward Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <main class="page-shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Duke FinTech 533 • Breakout Strategy Project</p>
        <h1>Single-stock breakout research report</h1>
        <p class="lede">I tested a fixed long-only breakout rule set on a liquid large-cap stock universe, used daily bars to define the setup, used hourly bars for the first clean breakout entry, and evaluated each candidate through rolling walk-forward out-of-sample windows. The strongest stock in this run was <strong>{focus_symbol}</strong>, which is the focus asset for the trade ledger and performance analysis below.</p>
      </div>
      <div class="hero-panel">
        <div class="hero-stat">
          <span>Selected asset</span>
          <strong>{focus_symbol}</strong>
        </div>
        <div class="hero-stat">
          <span>Out-of-sample trades</span>
          <strong>{len(focus_trades)}</strong>
        </div>
        <div class="hero-stat">
          <span>Total return</span>
          <strong>{pct(focus_summary["total_return"])}</strong>
        </div>
      </div>
    </section>

    <section class="content-grid">
      <article class="story-card">
        <h2>Strategy logic</h2>
        <p>The strategy now follows one fixed rule set instead of a parameter sweep. Daily bars define the setup: the breakout level is the highest high of the prior <strong>{focus_params["breakout_lookback"]} trading days</strong>, yesterday&apos;s close must be above the <strong>{focus_params["trend_window"]}-day {focus_params["trend_method"]}</strong>, and yesterday&apos;s volume must be at least <strong>{focus_params["min_volume_ratio"]:.0%}</strong> of its 20-day average. The intended execution logic is to buy the first clean breakout after that setup, risk <strong>1% of equity</strong>, place the stop at <strong>{focus_params["stop_atr"]:.1f} ATR</strong>, set the target at <strong>{focus_params["target_atr"]:.1f} ATR</strong>, and exit after <strong>{focus_params["max_hold_days"]} trading days</strong> if neither threshold is hit.</p>
      </article>

      <article class="story-card">
        <h2>Asset selection</h2>
        <p>{asset_selection_paragraph}</p>
      </article>

      <article class="story-card full-span">
        <h2>Breakout definition and assumptions</h2>
        <p>The breakout detection function is implemented in Python in <code>breakout_report_generator.py</code>. The parameters are explicit and easy to locate at the top of the script. The revised research design focuses on a single-stock universe instead of ETFs and combines quantitative evidence with qualitative judgment when picking the final candidate. Quantitatively, stocks are ranked by out-of-sample Sharpe ratio, out-of-sample return, drawdown, and trade count under the same fixed rule set. Qualitatively, I favor names with deep liquidity, durable institutional participation, and repeated breakout behavior while penalizing names whose gaps or event risk can overwhelm a clean trend-following setup. The backtest uses IBKR data through <code>shinybroker</code>, assumes <strong>2 bps slippage</strong> per side, <strong>$0.005 per share commissions</strong> with a <strong>$1 minimum</strong>, and sizes positions so that the initial stop risks about <strong>1% of current equity</strong>. The annual risk-free rate in the Sharpe ratio calculation is <strong>{RISK_FREE_RATE:.0%}</strong>.</p>
      </article>
    </section>

    <section class="metrics-section">
      <h2>Performance metrics</h2>
      <div class="metric-grid">
        {metrics_html}
      </div>
    </section>

    <section class="visual-grid">
      <figure class="visual-card">
        <img src="assets/equity_curve.png" alt="{focus_symbol} equity curve">
        <figcaption>Out-of-sample equity curve for the selected asset.</figcaption>
      </figure>
      <figure class="visual-card">
        <img src="assets/trade_outcomes.png" alt="{focus_symbol} trade outcomes">
        <figcaption>Trade outcome analysis: successful, timed out, or stop-loss triggered.</figcaption>
      </figure>
      <figure class="visual-card">
        <img src="assets/trade_pnl_histogram.png" alt="{focus_symbol} trade pnl histogram">
        <figcaption>Distribution of per-trade net PnL after commissions and slippage.</figcaption>
      </figure>
    </section>

    <section class="story-card full-span">
      <h2>Downloads</h2>
      <p>The full trade ledger is available as a CSV file, along with the window-level walk-forward summary and the asset ranking table.</p>
      <div class="download-row">
        <a href="data/{focus_symbol.lower()}_trade_blotter.csv">Download {focus_symbol} trade blotter</a>
        <a href="data/{focus_symbol.lower()}_walkforward_windows.csv">Download {focus_symbol} walk-forward windows</a>
        <a href="data/asset_ranking.csv">Download asset ranking</a>
      </div>
    </section>

    <section class="table-section">
      <div class="story-card">
        <h2>Stock ranking</h2>
        {dataframe_to_html(ranking_table, index=False)}
      </div>
      <div class="story-card">
        <h2>Walk-forward windows for {focus_symbol}</h2>
        {dataframe_to_html(top_windows, index=False)}
      </div>
      <div class="story-card full-span">
        <h2>Trade blotter for {focus_symbol}</h2>
        {dataframe_to_html(blotter, index=False)}
      </div>
      <div class="story-card">
        <h2>Outcome counts</h2>
        {dataframe_to_html(outcome_counts, index=False)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")


def write_css() -> None:
    css = """
:root {
  --bg: #f4efe7;
  --ink: #182028;
  --muted: #5f6b76;
  --panel: rgba(255, 252, 247, 0.88);
  --accent: #0b6e4f;
  --accent-soft: #dbeee4;
  --accent-warm: #d56b2d;
  --line: rgba(24, 32, 40, 0.12);
  --shadow: 0 18px 50px rgba(24, 32, 40, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Manrope", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(213, 107, 45, 0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(11, 110, 79, 0.18), transparent 32%),
    linear-gradient(180deg, #f7f3ec 0%, var(--bg) 100%);
}

.page-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 32px 0 48px;
}

.hero,
.story-card,
.visual-card,
.metric-card {
  background: var(--panel);
  backdrop-filter: blur(10px);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}

.hero {
  border-radius: 28px;
  padding: 32px;
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.eyebrow,
.metric-label,
code {
  font-family: "IBM Plex Mono", monospace;
}

.eyebrow {
  color: var(--accent);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

h1, h2 {
  margin-top: 0;
  line-height: 1.08;
}

h1 {
  font-size: clamp(2.2rem, 5vw, 4.4rem);
  margin-bottom: 12px;
}

.lede {
  font-size: 1.1rem;
  color: var(--muted);
  max-width: 70ch;
}

.hero-panel {
  display: grid;
  gap: 14px;
}

.hero-stat {
  border-radius: 22px;
  background: linear-gradient(135deg, var(--accent-soft), rgba(255,255,255,0.95));
  padding: 18px 20px;
}

.hero-stat span,
.metric-card p,
figcaption,
.story-card p {
  color: var(--muted);
}

.hero-stat strong {
  display: block;
  margin-top: 6px;
  font-size: 1.6rem;
}

.content-grid,
.visual-grid,
.table-section {
  display: grid;
  gap: 20px;
  margin-bottom: 24px;
}

.content-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.table-section {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.full-span {
  grid-column: 1 / -1;
}

.story-card,
.visual-card {
  border-radius: 24px;
  padding: 24px;
}

.metrics-section {
  margin-bottom: 24px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.metric-card {
  border-radius: 22px;
  padding: 20px;
}

.metric-value {
  font-size: 2rem;
  font-weight: 800;
  margin: 8px 0 10px;
}

.visual-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.visual-card img {
  width: 100%;
  border-radius: 16px;
  display: block;
}

.download-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.download-row a {
  display: inline-block;
  padding: 12px 16px;
  border-radius: 999px;
  background: var(--accent);
  color: white;
  text-decoration: none;
  font-weight: 700;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}

.data-table th,
.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

.data-table th {
  background: rgba(11, 110, 79, 0.08);
}

@media (max-width: 960px) {
  .hero,
  .content-grid,
  .metric-grid,
  .visual-grid,
  .table-section {
    grid-template-columns: 1fr;
  }
}
"""
    (ASSETS_DIR / "style.css").write_text(css.strip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_directories()

    price_data = {symbol: fetch_ibkr_price_data(symbol) for symbol in UNIVERSE}
    for symbol, data in price_data.items():
        data.daily.to_csv(DATA_DIR / f"{symbol.lower()}_daily_prices.csv", index=False)
        data.hourly.to_csv(DATA_DIR / f"{symbol.lower()}_hourly_prices.csv", index=False)

    walkforward_results = {
        symbol: run_walkforward_for_symbol(data, symbol) for symbol, data in price_data.items()
    }

    asset_rows = []
    for symbol, result in walkforward_results.items():
        summary = summarize_performance(result.trades, result.equity_curve)
        asset_rows.append(
            {
                "symbol": symbol,
                "windows": len(result.window_summary),
                "trade_count": len(result.trades),
                "avg_test_return": result.window_summary["total_return"].mean(),
                "avg_test_sharpe": result.window_summary["sharpe_ratio"].mean(),
                "avg_max_drawdown": result.window_summary["max_drawdown"].mean(),
                "total_return": summary["total_return"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "win_rate": summary["win_rate"],
            }
        )

        result.window_summary.to_csv(DATA_DIR / f"{symbol.lower()}_walkforward_windows.csv", index=False)
        result.parameter_history.to_csv(DATA_DIR / f"{symbol.lower()}_parameter_history.csv", index=False)
        result.trades.to_csv(DATA_DIR / f"{symbol.lower()}_trade_blotter.csv", index=False)
        result.equity_curve.to_csv(DATA_DIR / f"{symbol.lower()}_equity_curve.csv", index=False)

    asset_summary = pd.DataFrame(asset_rows).sort_values(
        ["avg_test_sharpe", "avg_test_return"], ascending=False
    ).reset_index(drop=True)
    asset_summary.to_csv(DATA_DIR / "asset_ranking.csv", index=False)

    focus_symbol = select_focus_asset(asset_summary)
    focus_result = walkforward_results[focus_symbol]
    focus_summary = summarize_performance(focus_result.trades, focus_result.equity_curve)
    focus_windows = focus_result.window_summary.copy().sort_values("test_start").reset_index(drop=True)

    if focus_windows.empty:
        raise RuntimeError(f"No walk-forward windows produced for {focus_symbol}.")

    most_common_params = (
        focus_windows.groupby(
            ["breakout_lookback", "trend_window", "trend_method", "atr_window", "stop_atr", "target_atr", "max_hold_days", "min_volume_ratio"],
            as_index=False,
        )
        .size()
        .sort_values("size", ascending=False)
        .iloc[0]
        .to_dict()
    )
    focus_params = {
        "breakout_lookback": int(most_common_params["breakout_lookback"]),
        "trend_window": int(most_common_params["trend_window"]),
        "trend_method": str(most_common_params["trend_method"]),
        "atr_window": int(most_common_params["atr_window"]),
        "stop_atr": float(most_common_params["stop_atr"]),
        "target_atr": float(most_common_params["target_atr"]),
        "max_hold_days": int(most_common_params["max_hold_days"]),
        "min_volume_ratio": float(most_common_params["min_volume_ratio"]),
    }

    focus_asset_row = asset_summary.loc[asset_summary["symbol"] == focus_symbol].iloc[0]
    qualitative_reason = QUALITATIVE_OVERLAY.get(focus_symbol, {}).get(
        "reason", "high liquidity and tradable trend structure"
    ).rstrip(".")
    asset_selection_paragraph = (
        f"I screened {len(asset_summary)} liquid large-cap stocks "
        f"({', '.join(asset_summary['symbol'].tolist())}) using the same fixed breakout rule set and the same rolling walk-forward framework with about one year of training data and the next quarter as out-of-sample test data. "
        f"Quantitatively, {focus_symbol} ranked best after combining average out-of-sample Sharpe ratio "
        f"({focus_asset_row['avg_test_sharpe']:.2f}), total out-of-sample return ({focus_asset_row['total_return']:.2%}), drawdown control, and trade count ({int(focus_asset_row['trade_count'])}). "
        f"Qualitatively, I preferred stocks with deep liquidity, sustained institutional participation, and a narrative that can support trend persistence without overwhelming the chart with one-off event risk. "
        f"For {focus_symbol}, that qualitative case was: {qualitative_reason}."
    )

    outcome_counts = (
        focus_result.trades["trade_outcome"]
        .value_counts()
        .rename_axis("trade_outcome")
        .reset_index(name="count")
    )
    outcome_counts.to_csv(DATA_DIR / f"{focus_symbol.lower()}_outcome_counts.csv", index=False)

    save_plot_images(focus_symbol, focus_result.equity_curve, focus_result.trades, outcome_counts)
    write_css()
    write_site(
        focus_symbol=focus_symbol,
        focus_params=focus_params,
        focus_summary=focus_summary,
        asset_summary=asset_summary,
        focus_trades=focus_result.trades.copy(),
        focus_windows=focus_windows,
        outcome_counts=outcome_counts,
        asset_selection_paragraph=asset_selection_paragraph,
    )

    print(f"Generated report for focus asset: {focus_symbol}")
    print(f"IBKR connection used: host={IB_HOST}, port={IB_PORT}, client_id={IB_CLIENT_ID}")
    print(f"Site: {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
