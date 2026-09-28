# ============================================================
# backtest/trading_engine.py
#
# TradingBacktestEngine v5.1
#
# main.py v4 対応
# ai/model.py v2 対応
# backtest/engine.py v2 対応
#
# ------------------------------------------------------------
# 売買タイミング
#
# t日 終値:
#   BUY / SELL シグナル判定
#
# t+1日 始値:
#   約定
#
# BUY約定当日:
#   SELL判定しない
#
# ------------------------------------------------------------
# 会計
#
# ・買い手数料
# ・売り手数料
# ・買いスリッページ
# ・売りスリッページ
# ・純損益
# ・最終資産
# ・会計整合性
#
# を明示的に計算
# ============================================================


import numpy as np
import pandas as pd

from strategy.entry import EntryStrategy
from strategy.exit import ExitStrategy
from strategy.risk import RiskManager


# ============================================================
# TradingBacktestEngine
# ============================================================

class TradingBacktestEngine:

    def __init__(
        self,

        initial_capital=1_000_000,

        lot_size=100,

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        entry_minimum_score=6,

        entry_minimum_probability=0.55,

        entry_strong_probability=0.60,

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        stop_loss_rate=0.05,

        take_profit_rate=0.10,

        ai_exit_probability=0.45,

        max_holding_days=10,

        minimum_exit_score=3,

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_per_trade=0.01,

        max_position_rate=0.50,

        # ----------------------------------------------------
        # Cost
        # ----------------------------------------------------

        commission_rate=0.001,

        slippage_rate=0.001,
    ):

        # ====================================================
        # Basic
        # ====================================================

        self.initial_capital = float(
            initial_capital
        )

        self.lot_size = int(
            lot_size
        )


        # ====================================================
        # Entry settings
        # ====================================================

        self.entry_minimum_score = int(
            entry_minimum_score
        )

        self.entry_minimum_probability = float(
            entry_minimum_probability
        )

        self.entry_strong_probability = float(
            entry_strong_probability
        )


        # ====================================================
        # Exit settings
        # ====================================================

        self.stop_loss_rate = float(
            stop_loss_rate
        )

        self.take_profit_rate = float(
            take_profit_rate
        )

        self.ai_exit_probability = float(
            ai_exit_probability
        )

        self.max_holding_days = int(
            max_holding_days
        )

        self.minimum_exit_score = int(
            minimum_exit_score
        )


        # ====================================================
        # Risk settings
        # ====================================================

        self.risk_per_trade = float(
            risk_per_trade
        )

        self.max_position_rate = float(
            max_position_rate
        )


        # ====================================================
        # Costs
        # ====================================================

        self.commission_rate = float(
            commission_rate
        )

        self.slippage_rate = float(
            slippage_rate
        )


        # ====================================================
        # Strategies
        # ====================================================

        self.entry_strategy = EntryStrategy(

            minimum_score=
                self.entry_minimum_score,

            minimum_probability=
                self.entry_minimum_probability,

            strong_probability=
                self.entry_strong_probability,
        )


        self.exit_strategy = ExitStrategy(

            stop_loss_rate=
                self.stop_loss_rate,

            take_profit_rate=
                self.take_profit_rate,

            ai_exit_probability=
                self.ai_exit_probability,

            max_holding_days=
                self.max_holding_days,

            minimum_exit_score=
                self.minimum_exit_score,
        )


        self.risk_manager = RiskManager(

            lot_size=
                self.lot_size,

            risk_per_trade=
                self.risk_per_trade,

            max_position_rate=
                self.max_position_rate,

            stop_loss_rate=
                self.stop_loss_rate,

            take_profit_rate=
                self.take_profit_rate,

            commission_rate=
                self.commission_rate,

            slippage_rate=
                self.slippage_rate,
        )


        # ====================================================
        # Results
        # ====================================================

        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.skipped_entries = pd.DataFrame()

        self.order_log = pd.DataFrame()

        self.metrics = {}


    # ========================================================
    # Safe float
    # ========================================================

    @staticmethod
    def _safe_float(
        value,
        default=None,
    ):

        try:

            if value is None:
                return default


            if isinstance(
                value,
                pd.DataFrame,
            ):

                if value.empty:
                    return default

                value = value.iloc[
                    -1,
                    -1,
                ]


            if isinstance(
                value,
                pd.Series,
            ):

                values = pd.to_numeric(
                    value,
                    errors="coerce",
                ).dropna()

                if values.empty:
                    return default

                value = values.iloc[-1]


            if isinstance(
                value,
                (
                    np.ndarray,
                    list,
                    tuple,
                ),
            ):

                array = np.asarray(
                    value
                ).reshape(-1)

                if len(array) == 0:
                    return default

                value = array[-1]


            result = float(
                value
            )


            if not np.isfinite(
                result
            ):

                return default


            return result


        except Exception:

            return default


    # ========================================================
    # Date normalization
    # ========================================================

    @staticmethod
    def _normalize_index(
        dataframe,
    ):

        if dataframe is None:

            return pd.DataFrame()


        result = dataframe.copy()


        if result.empty:

            return result


        result.index = pd.to_datetime(
            result.index
        )


        if getattr(
            result.index,
            "tz",
            None,
        ) is not None:

            result.index = (
                result.index
                .tz_localize(None)
            )


        result = result[
            ~result.index.duplicated(
                keep="last"
            )
        ]


        result = result.sort_index()


        return result


    # ========================================================
    # Entry strategy wrapper
    # ========================================================

    def _evaluate_entry(
        self,
        row,
        probability,
    ):

        try:

            result = (
                self.entry_strategy.evaluate(
                    row=row,
                    probability=probability,
                )
            )

        except TypeError:

            try:

                result = (
                    self.entry_strategy.evaluate(
                        row,
                        probability,
                    )
                )

            except Exception:

                return {
                    "action": "HOLD",
                    "score": 0,
                    "reason": "ENTRY_ERROR",
                }

        except Exception:

            return {
                "action": "HOLD",
                "score": 0,
                "reason": "ENTRY_ERROR",
            }


        if result is None:

            return {
                "action": "HOLD",
                "score": 0,
            }


        if isinstance(
            result,
            str,
        ):

            return {
                "action": result,
                "score": 0,
            }


        if not isinstance(
            result,
            dict,
        ):

            return {
                "action": "HOLD",
                "score": 0,
            }


        return result


    # ========================================================
    # Exit strategy wrapper
    # ========================================================

    def _evaluate_exit(
        self,
        row,
        probability,
        entry_price,
        holding_days,
    ):

        try:

            result = (
                self.exit_strategy.evaluate(

                    row=row,

                    probability=
                        probability,

                    entry_price=
                        entry_price,

                    holding_days=
                        holding_days,
                )
            )

        except TypeError:

            try:

                result = (
                    self.exit_strategy.evaluate(

                        row,

                        probability,

                        entry_price,

                        holding_days,
                    )
                )

            except Exception:

                result = None

        except Exception:

            result = None


        # ====================================================
        # ExitStrategyで評価できなかった場合
        # 最低限のハードExitを使用
        # ====================================================

        if result is None:

            close_price = (
                self._safe_float(
                    row.get(
                        "Close"
                    )
                )
            )


            if (
                close_price is None
                or
                entry_price <= 0
            ):

                return {
                    "action": "HOLD",
                    "score": 0,
                    "reason": "NO_CLOSE",
                }


            return_rate = (

                close_price
                / entry_price
                - 1.0
            )


            if (
                return_rate
                <= -self.stop_loss_rate
            ):

                return {
                    "action": "SELL",
                    "score": 99,
                    "reason": "STOP_LOSS",
                }


            if (
                return_rate
                >= self.take_profit_rate
            ):

                return {
                    "action": "SELL",
                    "score": 99,
                    "reason": "TAKE_PROFIT",
                }


            if (
                holding_days
                >= self.max_holding_days
            ):

                return {
                    "action": "SELL",
                    "score": 99,
                    "reason": "MAX_HOLDING",
                }


            if (
                probability
                <= self.ai_exit_probability
            ):

                return {
                    "action": "SELL",
                    "score": 99,
                    "reason": "AI_EXIT",
                }


            return {
                "action": "HOLD",
                "score": 0,
                "reason": "HOLD",
            }


        if isinstance(
            result,
            str,
        ):

            return {
                "action": result,
                "score": 0,
            }


        if not isinstance(
            result,
            dict,
        ):

            return {
                "action": "HOLD",
                "score": 0,
            }


        return result


    # ========================================================
    # Risk manager wrapper
    # ========================================================

    def _evaluate_risk(
        self,
        capital,
        market_price,
    ):

        try:

            result = (
                self.risk_manager.evaluate_trade(

                    capital=
                        capital,

                    market_price=
                        market_price,
                )
            )

        except TypeError:

            try:

                result = (
                    self.risk_manager
                    .evaluate_trade(
                        capital,
                        market_price,
                    )
                )

            except Exception:

                result = None

        except Exception:

            result = None


        if not isinstance(
            result,
            dict,
        ):

            return {
                "can_trade": False,
                "shares": 0,
                "reason": "RISK_ERROR",
            }


        return result


    # ========================================================
    # Shares extraction
    # ========================================================

    def _extract_shares(
        self,
        risk_result,
    ):

        possible_keys = [

            "shares",
            "share_count",
            "quantity",
            "position_size",
        ]


        shares = None


        for key in possible_keys:

            if key in risk_result:

                shares = (
                    self._safe_float(
                        risk_result.get(
                            key
                        )
                    )
                )

                if shares is not None:
                    break


        if shares is None:

            return 0


        shares = int(
            shares
        )


        if shares <= 0:

            return 0


        # ====================================================
        # 売買単位へ丸める
        # ====================================================

        shares = (

            shares
            // self.lot_size

            * self.lot_size
        )


        return max(
            shares,
            0,
        )


    # ========================================================
    # Main backtest
    # ========================================================

    def run(
        self,
        stock_data,
        ai_data,
        walk_results,
    ):

        # ====================================================
        # Normalize
        # ====================================================

        stock = self._normalize_index(
            stock_data
        )

        ai = self._normalize_index(
            ai_data
        )

        walk = self._normalize_index(
            walk_results
        )


        # ====================================================
        # Validation
        # ====================================================

        if stock.empty:

            raise ValueError(
                "stock_data が空です。"
            )


        if ai.empty:

            raise ValueError(
                "ai_data が空です。"
            )


        if walk.empty:

            raise ValueError(
                "walk_results が空です。"
            )


        for column in [
            "Open",
            "Close",
        ]:

            if column not in stock.columns:

                raise ValueError(
                    f"stock_data に "
                    f"{column} 列がありません。"
                )


        if (
            "Probability_Up"
            not in walk.columns
        ):

            raise ValueError(
                "walk_results に "
                "Probability_Up 列がありません。"
            )


        # ====================================================
        # Numeric
        # ====================================================

        stock[
            "Open"
        ] = pd.to_numeric(
            stock[
                "Open"
            ],
            errors="coerce",
        )


        stock[
            "Close"
        ] = pd.to_numeric(
            stock[
                "Close"
            ],
            errors="coerce",
        )


        walk[
            "Probability_Up"
        ] = pd.to_numeric(
            walk[
                "Probability_Up"
            ],
            errors="coerce",
        )


        # ====================================================
        # Reset
        # ====================================================

        cash = float(
            self.initial_capital
        )


        position_shares = 0

        entry_price = None

        entry_market_open = None

        entry_date = None

        entry_signal_date = None

        entry_probability = None

        entry_score = None

        entry_commission = 0.0

        entry_slippage_cost = 0.0


        pending_order = None


        trades = []

        equity_records = []

        skipped_entries = []

        order_records = []


        total_commission = 0.0

        total_slippage = 0.0


        # ====================================================
        # Date positions
        # ====================================================

        stock_dates = list(
            stock.index
        )


        date_to_position = {

            date: index

            for index, date
            in enumerate(
                stock_dates
            )
        }


        # ====================================================
        # Loop
        # ====================================================

        for date_position, date in enumerate(
            stock_dates
        ):

            stock_row = (
                stock.loc[
                    date
                ]
            )


            market_open = (
                self._safe_float(
                    stock_row.get(
                        "Open"
                    )
                )
            )


            market_close = (
                self._safe_float(
                    stock_row.get(
                        "Close"
                    )
                )
            )


            if (
                market_open is None
                or
                market_close is None
                or
                market_open <= 0
                or
                market_close <= 0
            ):

                continue


            # =================================================
            # BUYした日か
            # =================================================

            bought_today = False


            # =================================================
            # 1. Pending order を当日始値で約定
            # =================================================

            if pending_order is not None:

                order_type = (
                    pending_order[
                        "type"
                    ]
                )


                # =============================================
                # BUY fill
                # =============================================

                if (
                    order_type == "BUY"
                    and
                    position_shares == 0
                ):

                    # -----------------------------------------
                    # 買いは不利な方向へスリッページ
                    # -----------------------------------------

                    execution_price = (

                        market_open

                        * (
                            1.0
                            + self.slippage_rate
                        )
                    )


                    risk_result = (
                        self._evaluate_risk(

                            capital=cash,

                            market_price=
                                execution_price,
                        )
                    )


                    can_trade = bool(
                        risk_result.get(
                            "can_trade",
                            False,
                        )
                    )


                    shares = (
                        self._extract_shares(
                            risk_result
                        )
                    )


                    if (
                        can_trade
                        and
                        shares >= self.lot_size
                    ):

                        gross_purchase = (

                            execution_price
                            * shares
                        )


                        buy_commission = (

                            gross_purchase
                            * self.commission_rate
                        )


                        required_cash = (

                            gross_purchase
                            + buy_commission
                        )


                        # =====================================
                        # 念のため実際のcashでも確認
                        # =====================================

                        if required_cash <= cash:

                            buy_slippage_cost = (

                                (
                                    execution_price
                                    - market_open
                                )

                                * shares
                            )


                            cash -= required_cash


                            position_shares = (
                                shares
                            )


                            entry_price = float(
                                execution_price
                            )


                            entry_market_open = float(
                                market_open
                            )


                            entry_date = date


                            entry_signal_date = (
                                pending_order[
                                    "signal_date"
                                ]
                            )


                            entry_probability = (
                                pending_order[
                                    "probability"
                                ]
                            )


                            entry_score = (
                                pending_order[
                                    "score"
                                ]
                            )


                            entry_commission = float(
                                buy_commission
                            )


                            entry_slippage_cost = float(
                                buy_slippage_cost
                            )


                            total_commission += (
                                buy_commission
                            )


                            total_slippage += (
                                buy_slippage_cost
                            )


                            bought_today = True


                            order_records.append({

                                "Signal_Date":
                                    entry_signal_date,

                                "Execution_Date":
                                    date,

                                "Order":
                                    "BUY",

                                "Status":
                                    "FILLED",

                                "Market_Price":
                                    market_open,

                                "Execution_Price":
                                    execution_price,

                                "Shares":
                                    shares,

                                "Commission":
                                    buy_commission,

                                "Slippage_Cost":
                                    buy_slippage_cost,

                                "Probability":
                                    entry_probability,

                                "Score":
                                    entry_score,
                            })


                        else:

                            skipped_entries.append({

                                "Date":
                                    date,

                                "Signal_Date":
                                    pending_order[
                                        "signal_date"
                                    ],

                                "Reason":
                                    "INSUFFICIENT_CASH",

                                "Market_Open":
                                    market_open,

                                "Execution_Price":
                                    execution_price,

                                "Cash":
                                    cash,

                                "Shares":
                                    shares,
                            })


                            order_records.append({

                                "Signal_Date":
                                    pending_order[
                                        "signal_date"
                                    ],

                                "Execution_Date":
                                    date,

                                "Order":
                                    "BUY",

                                "Status":
                                    "REJECTED_CASH",

                                "Market_Price":
                                    market_open,

                                "Execution_Price":
                                    execution_price,

                                "Shares":
                                    shares,
                            })


                    else:

                        reason = (
                            risk_result.get(
                                "reason",
                                risk_result.get(
                                    "message",
                                    "RISK_REJECTED",
                                ),
                            )
                        )


                        skipped_entries.append({

                            "Date":
                                date,

                            "Signal_Date":
                                pending_order[
                                    "signal_date"
                                ],

                            "Reason":
                                reason,

                            "Market_Open":
                                market_open,

                            "Cash":
                                cash,

                            "Probability":
                                pending_order[
                                    "probability"
                                ],

                            "Score":
                                pending_order[
                                    "score"
                                ],
                        })


                        order_records.append({

                            "Signal_Date":
                                pending_order[
                                    "signal_date"
                                ],

                            "Execution_Date":
                                date,

                            "Order":
                                "BUY",

                            "Status":
                                "REJECTED_RISK",

                            "Market_Price":
                                market_open,

                            "Shares":
                                shares,

                            "Reason":
                                reason,
                        })


                    pending_order = None


                # =============================================
                # SELL fill
                # =============================================

                elif (
                    order_type == "SELL"
                    and
                    position_shares > 0
                ):

                    # -----------------------------------------
                    # 売りは不利な方向へスリッページ
                    # -----------------------------------------

                    execution_price = (

                        market_open

                        * (
                            1.0
                            - self.slippage_rate
                        )
                    )


                    shares = int(
                        position_shares
                    )


                    gross_sale = (

                        execution_price
                        * shares
                    )


                    sell_commission = (

                        gross_sale
                        * self.commission_rate
                    )


                    sell_slippage_cost = (

                        (
                            market_open
                            - execution_price
                        )

                        * shares
                    )


                    cash += (

                        gross_sale
                        - sell_commission
                    )


                    total_commission += (
                        sell_commission
                    )


                    total_slippage += (
                        sell_slippage_cost
                    )


                    # =========================================
                    # Trade P&L
                    # =========================================

                    gross_execution_pnl = (

                        (
                            execution_price
                            - entry_price
                        )

                        * shares
                    )


                    net_profit = (

                        gross_execution_pnl

                        - entry_commission

                        - sell_commission
                    )


                    profit_rate = (

                        net_profit

                        /

                        (
                            entry_price
                            * shares
                            + entry_commission
                        )
                    )


                    # =========================================
                    # Market-price P&L audit
                    #
                    # スリッページ前の始値同士
                    # =========================================

                    market_gross_pnl = (

                        (
                            market_open
                            - entry_market_open
                        )

                        * shares
                    )


                    total_trade_slippage = (

                        entry_slippage_cost

                        + sell_slippage_cost
                    )


                    total_trade_commission = (

                        entry_commission

                        + sell_commission
                    )


                    audit_net_profit = (

                        market_gross_pnl

                        - total_trade_slippage

                        - total_trade_commission
                    )


                    audit_difference = (

                        net_profit

                        - audit_net_profit
                    )


                    # =========================================
                    # Holding days
                    # =========================================

                    if (
                        entry_date
                        in date_to_position
                    ):

                        holding_days = (

                            date_position

                            - date_to_position[
                                entry_date
                            ]
                        )

                    else:

                        holding_days = 0


                    trades.append({

                        "Entry_Signal_Date":
                            entry_signal_date,

                        "Entry_Date":
                            entry_date,

                        "Exit_Signal_Date":
                            pending_order[
                                "signal_date"
                            ],

                        "Exit_Date":
                            date,

                        "Entry_Market_Open":
                            entry_market_open,

                        "Entry_Price":
                            entry_price,

                        "Exit_Market_Open":
                            market_open,

                        "Exit_Price":
                            execution_price,

                        "Shares":
                            shares,

                        "Holding_Days":
                            holding_days,

                        "Entry_Probability":
                            entry_probability,

                        "Entry_Score":
                            entry_score,

                        "Exit_Probability":
                            pending_order[
                                "probability"
                            ],

                        "Exit_Score":
                            pending_order[
                                "score"
                            ],

                        "Exit_Reason":
                            pending_order[
                                "reason"
                            ],

                        "Entry_Commission":
                            entry_commission,

                        "Exit_Commission":
                            sell_commission,

                        "Total_Commission":
                            total_trade_commission,

                        "Entry_Slippage":
                            entry_slippage_cost,

                        "Exit_Slippage":
                            sell_slippage_cost,

                        "Total_Slippage":
                            total_trade_slippage,

                        "Market_Gross_PnL":
                            market_gross_pnl,

                        "Execution_Gross_PnL":
                            gross_execution_pnl,

                        "Net_Profit":
                            net_profit,

                        "Profit_Rate":
                            profit_rate,

                        "Audit_Net_Profit":
                            audit_net_profit,

                        "Audit_Difference":
                            audit_difference,

                        "Forced_Exit":
                            False,
                    })


                    order_records.append({

                        "Signal_Date":
                            pending_order[
                                "signal_date"
                            ],

                        "Execution_Date":
                            date,

                        "Order":
                            "SELL",

                        "Status":
                            "FILLED",

                        "Market_Price":
                            market_open,

                        "Execution_Price":
                            execution_price,

                        "Shares":
                            shares,

                        "Commission":
                            sell_commission,

                        "Slippage_Cost":
                            sell_slippage_cost,

                        "Probability":
                            pending_order[
                                "probability"
                            ],

                        "Score":
                            pending_order[
                                "score"
                            ],

                        "Reason":
                            pending_order[
                                "reason"
                            ],
                    })


                    # =========================================
                    # Position reset
                    # =========================================

                    position_shares = 0

                    entry_price = None

                    entry_market_open = None

                    entry_date = None

                    entry_signal_date = None

                    entry_probability = None

                    entry_score = None

                    entry_commission = 0.0

                    entry_slippage_cost = 0.0

                    pending_order = None


                else:

                    # -----------------------------------------
                    # 状態不一致なら注文を破棄
                    # -----------------------------------------

                    pending_order = None


            # =================================================
            # 2. Equity at close
            # =================================================

            position_value = (

                position_shares
                * market_close
            )


            total_equity = (

                cash
                + position_value
            )


            equity_records.append({

                "Date":
                    date,

                "Cash":
                    cash,

                "Position_Value":
                    position_value,

                "Total_Equity":
                    total_equity,

                "Shares":
                    position_shares,

                "Close":
                    market_close,
            })


            # =================================================
            # 3. Walk-forward probability
            #
            # この日に予測がなければ
            # シグナル判定しない
            # =================================================

            if date not in walk.index:

                continue


            probability = (
                self._safe_float(

                    walk.loc[
                        date,
                        "Probability_Up"
                    ]
                )
            )


            if probability is None:

                continue


            # =================================================
            # AI feature row
            # =================================================

            if date not in ai.index:

                continue


            feature_row = (
                ai.loc[
                    date
                ]
            )


            # =================================================
            # 4. Positionなし → BUY signal
            # =================================================

            if (
                position_shares == 0
                and
                pending_order is None
            ):

                entry_result = (
                    self._evaluate_entry(

                        row=
                            feature_row,

                        probability=
                            probability,
                    )
                )


                action = str(
                    entry_result.get(
                        "action",
                        "HOLD",
                    )
                ).upper()


                score = int(
                    self._safe_float(
                        entry_result.get(
                            "score"
                        ),
                        0,
                    )
                )


                if action == "BUY":

                    # -----------------------------------------
                    # 翌営業日が存在する場合だけ注文
                    # -----------------------------------------

                    if (
                        date_position
                        < len(stock_dates) - 1
                    ):

                        pending_order = {

                            "type":
                                "BUY",

                            "signal_date":
                                date,

                            "probability":
                                probability,

                            "score":
                                score,

                            "reason":
                                entry_result.get(
                                    "reason",
                                    entry_result.get(
                                        "reasons",
                                        "ENTRY_SIGNAL",
                                    ),
                                ),
                        }


                        order_records.append({

                            "Signal_Date":
                                date,

                            "Execution_Date":
                                stock_dates[
                                    date_position
                                    + 1
                                ],

                            "Order":
                                "BUY",

                            "Status":
                                "PENDING",

                            "Probability":
                                probability,

                            "Score":
                                score,
                        })


            # =================================================
            # 5. Positionあり → SELL signal
            #
            # BUY約定当日は絶対にSELL判定しない
            # =================================================

            elif (
                position_shares > 0
                and
                pending_order is None
                and
                not bought_today
            ):

                if (
                    entry_date
                    in date_to_position
                ):

                    holding_days = (

                        date_position

                        - date_to_position[
                            entry_date
                        ]
                    )

                else:

                    holding_days = 0


                # =============================================
                # Entry day = 0
                #
                # 翌営業日の終値から
                # SELL評価開始
                # =============================================

                if holding_days >= 1:

                    exit_result = (
                        self._evaluate_exit(

                            row=
                                feature_row,

                            probability=
                                probability,

                            entry_price=
                                entry_price,

                            holding_days=
                                holding_days,
                        )
                    )


                    action = str(
                        exit_result.get(
                            "action",
                            "HOLD",
                        )
                    ).upper()


                    score = int(
                        self._safe_float(
                            exit_result.get(
                                "score"
                            ),
                            0,
                        )
                    )


                    if action == "SELL":

                        if (
                            date_position
                            < len(
                                stock_dates
                            ) - 1
                        ):

                            pending_order = {

                                "type":
                                    "SELL",

                                "signal_date":
                                    date,

                                "probability":
                                    probability,

                                "score":
                                    score,

                                "reason":
                                    exit_result.get(
                                        "reason",
                                        exit_result.get(
                                            "reasons",
                                            "EXIT_SIGNAL",
                                        ),
                                    ),
                            }


                            order_records.append({

                                "Signal_Date":
                                    date,

                                "Execution_Date":
                                    stock_dates[
                                        date_position
                                        + 1
                                    ],

                                "Order":
                                    "SELL",

                                "Status":
                                    "PENDING",

                                "Probability":
                                    probability,

                                "Score":
                                    score,

                                "Reason":
                                    pending_order[
                                        "reason"
                                    ],
                            })


        # ====================================================
        # 最終日にポジションが残った場合
        #
        # データ終了のため最終Closeで強制決済
        # ====================================================

        if (
            position_shares > 0
            and
            len(stock_dates) > 0
        ):

            final_date = (
                stock_dates[-1]
            )


            final_row = (
                stock.loc[
                    final_date
                ]
            )


            final_close = (
                self._safe_float(
                    final_row.get(
                        "Close"
                    )
                )
            )


            if (
                final_close is not None
                and
                final_close > 0
            ):

                # ---------------------------------------------
                # 売りスリッページ
                # ---------------------------------------------

                execution_price = (

                    final_close

                    * (
                        1.0
                        - self.slippage_rate
                    )
                )


                shares = int(
                    position_shares
                )


                gross_sale = (

                    execution_price
                    * shares
                )


                sell_commission = (

                    gross_sale
                    * self.commission_rate
                )


                sell_slippage_cost = (

                    (
                        final_close
                        - execution_price
                    )

                    * shares
                )


                cash += (

                    gross_sale
                    - sell_commission
                )


                total_commission += (
                    sell_commission
                )


                total_slippage += (
                    sell_slippage_cost
                )


                gross_execution_pnl = (

                    (
                        execution_price
                        - entry_price
                    )

                    * shares
                )


                net_profit = (

                    gross_execution_pnl

                    - entry_commission

                    - sell_commission
                )


                profit_rate = (

                    net_profit

                    /

                    (
                        entry_price
                        * shares
                        + entry_commission
                    )
                )


                # ---------------------------------------------
                # Audit
                # ---------------------------------------------

                market_gross_pnl = (

                    (
                        final_close
                        - entry_market_open
                    )

                    * shares
                )


                total_trade_slippage = (

                    entry_slippage_cost

                    + sell_slippage_cost
                )


                total_trade_commission = (

                    entry_commission

                    + sell_commission
                )


                audit_net_profit = (

                    market_gross_pnl

                    - total_trade_slippage

                    - total_trade_commission
                )


                audit_difference = (

                    net_profit

                    - audit_net_profit
                )


                final_position = (
                    len(stock_dates) - 1
                )


                entry_position = (
                    date_to_position.get(
                        entry_date,
                        final_position,
                    )
                )


                holding_days = max(
                    0,
                    final_position
                    - entry_position,
                )


                trades.append({

                    "Entry_Signal_Date":
                        entry_signal_date,

                    "Entry_Date":
                        entry_date,

                    "Exit_Signal_Date":
                        final_date,

                    "Exit_Date":
                        final_date,

                    "Entry_Market_Open":
                        entry_market_open,

                    "Entry_Price":
                        entry_price,

                    "Exit_Market_Open":
                        final_close,

                    "Exit_Price":
                        execution_price,

                    "Shares":
                        shares,

                    "Holding_Days":
                        holding_days,

                    "Entry_Probability":
                        entry_probability,

                    "Entry_Score":
                        entry_score,

                    "Exit_Probability":
                        np.nan,

                    "Exit_Score":
                        np.nan,

                    "Exit_Reason":
                        "FINAL_FORCED_EXIT",

                    "Entry_Commission":
                        entry_commission,

                    "Exit_Commission":
                        sell_commission,

                    "Total_Commission":
                        total_trade_commission,

                    "Entry_Slippage":
                        entry_slippage_cost,

                    "Exit_Slippage":
                        sell_slippage_cost,

                    "Total_Slippage":
                        total_trade_slippage,

                    "Market_Gross_PnL":
                        market_gross_pnl,

                    "Execution_Gross_PnL":
                        gross_execution_pnl,

                    "Net_Profit":
                        net_profit,

                    "Profit_Rate":
                        profit_rate,

                    "Audit_Net_Profit":
                        audit_net_profit,

                    "Audit_Difference":
                        audit_difference,

                    "Forced_Exit":
                        True,
                })


                order_records.append({

                    "Signal_Date":
                        final_date,

                    "Execution_Date":
                        final_date,

                    "Order":
                        "SELL",

                    "Status":
                        "FORCED_FILLED",

                    "Market_Price":
                        final_close,

                    "Execution_Price":
                        execution_price,

                    "Shares":
                        shares,

                    "Commission":
                        sell_commission,

                    "Slippage_Cost":
                        sell_slippage_cost,

                    "Reason":
                        "FINAL_FORCED_EXIT",
                })


                position_shares = 0


                # =============================================
                # 最終Equityを決済後cashへ修正
                # =============================================

                if equity_records:

                    equity_records[
                        -1
                    ][
                        "Cash"
                    ] = cash

                    equity_records[
                        -1
                    ][
                        "Position_Value"
                    ] = 0.0

                    equity_records[
                        -1
                    ][
                        "Total_Equity"
                    ] = cash

                    equity_records[
                        -1
                    ][
                        "Shares"
                    ] = 0


        # ====================================================
        # DataFrames
        # ====================================================

        self.trades = pd.DataFrame(
            trades
        )


        self.skipped_entries = (
            pd.DataFrame(
                skipped_entries
            )
        )


        self.order_log = (
            pd.DataFrame(
                order_records
            )
        )


        self.equity_curve = (
            pd.DataFrame(
                equity_records
            )
        )


        if (
            not self.equity_curve.empty
            and
            "Date"
            in self.equity_curve.columns
        ):

            self.equity_curve[
                "Date"
            ] = pd.to_datetime(
                self.equity_curve[
                    "Date"
                ]
            )


            self.equity_curve = (
                self.equity_curve
                .drop_duplicates(
                    subset=[
                        "Date"
                    ],
                    keep="last",
                )
                .set_index(
                    "Date"
                )
                .sort_index()
            )


        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = (
            self._calculate_metrics(

                final_cash=cash,

                total_commission=
                    total_commission,

                total_slippage=
                    total_slippage,
            )
        )


        return (
            self.trades.copy(),
            self.equity_curve.copy(),
            dict(
                self.metrics
            ),
        )


    # ========================================================
    # Metrics
    # ========================================================

    def _calculate_metrics(
        self,
        final_cash,
        total_commission,
        total_slippage,
    ):

        final_capital = float(
            final_cash
        )


        total_profit = (

            final_capital

            - self.initial_capital
        )


        total_return = (

            total_profit
            / self.initial_capital

            if self.initial_capital != 0

            else 0.0
        )


        # ====================================================
        # No trades
        # ====================================================

        if self.trades.empty:

            trade_count = 0

            wins = 0

            losses = 0

            flat = 0

            win_rate = None

            average_profit = None

            average_profit_rate = None

            gross_profit = 0.0

            gross_loss = 0.0

            best_trade = None

            worst_trade = None

            profit_factor = None

            average_holding_days = None

            trade_profit_sum = 0.0

            trade_audit_difference = 0.0


        else:

            pnl = pd.to_numeric(
                self.trades[
                    "Net_Profit"
                ],
                errors="coerce",
            ).fillna(0.0)


            profit_rate = pd.to_numeric(
                self.trades[
                    "Profit_Rate"
                ],
                errors="coerce",
            )


            holding = pd.to_numeric(
                self.trades[
                    "Holding_Days"
                ],
                errors="coerce",
            )


            trade_count = int(
                len(
                    self.trades
                )
            )


            wins = int(
                (
                    pnl > 0
                ).sum()
            )


            losses = int(
                (
                    pnl < 0
                ).sum()
            )


            flat = int(
                (
                    pnl == 0
                ).sum()
            )


            win_rate = (

                wins / trade_count

                if trade_count > 0

                else None
            )


            average_profit = float(
                pnl.mean()
            )


            average_profit_rate = (

                float(
                    profit_rate.mean()
                )

                if profit_rate.notna().any()

                else None
            )


            gross_profit = float(
                pnl[
                    pnl > 0
                ].sum()
            )


            # ------------------------------------------------
            # 正の損失額として保存
            # main.py側で - を付けて表示
            # ------------------------------------------------

            gross_loss = float(
                -pnl[
                    pnl < 0
                ].sum()
            )


            best_trade = float(
                pnl.max()
            )


            worst_trade = float(
                pnl.min()
            )


            if gross_loss > 0:

                profit_factor = (

                    gross_profit
                    / gross_loss
                )

            elif gross_profit > 0:

                profit_factor = float(
                    "inf"
                )

            else:

                profit_factor = None


            average_holding_days = (

                float(
                    holding.mean()
                )

                if holding.notna().any()

                else None
            )


            trade_profit_sum = float(
                pnl.sum()
            )


            if (
                "Audit_Difference"
                in self.trades.columns
            ):

                trade_audit_difference = float(

                    pd.to_numeric(
                        self.trades[
                            "Audit_Difference"
                        ],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .sum()
                )

            else:

                trade_audit_difference = 0.0


        # ====================================================
        # Drawdown
        # ====================================================

        max_drawdown = 0.0


        if (
            not self.equity_curve.empty
            and
            "Total_Equity"
            in self.equity_curve.columns
        ):

            equity = pd.to_numeric(
                self.equity_curve[
                    "Total_Equity"
                ],
                errors="coerce",
            ).dropna()


            if not equity.empty:

                running_max = (
                    equity.cummax()
                )


                drawdown = (

                    equity
                    / running_max
                    - 1.0
                )


                max_drawdown = abs(
                    float(
                        drawdown.min()
                    )
                )


        # ====================================================
        # Accounting
        #
        # ① 最終資産 - 初期資産
        # ② 全取引純損益合計
        #
        # が一致するか
        # ====================================================

        accounting_difference = (

            total_profit

            - trade_profit_sum
        )


        accounting_tolerance = 0.01


        accounting_ok = (

            abs(
                accounting_difference
            )
            <= accounting_tolerance

            and

            abs(
                trade_audit_difference
            )
            <= accounting_tolerance
        )


        total_trading_cost = (

            float(
                total_commission
            )

            +

            float(
                total_slippage
            )
        )


        return {

            # ================================================
            # Version
            # ================================================

            "engine_version":
                "v5.1",

            # ================================================
            # Capital
            # ================================================

            "initial_capital":
                float(
                    self.initial_capital
                ),

            "final_capital":
                final_capital,

            "total_profit":
                float(
                    total_profit
                ),

            "total_return":
                float(
                    total_return
                ),

            # ================================================
            # Trades
            # ================================================

            "trade_count":
                int(
                    trade_count
                ),

            "trades":
                int(
                    trade_count
                ),

            "wins":
                int(
                    wins
                ),

            "losses":
                int(
                    losses
                ),

            "flat":
                int(
                    flat
                ),

            "win_rate":
                win_rate,

            "average_profit":
                average_profit,

            "average_profit_rate":
                average_profit_rate,

            "gross_profit":
                float(
                    gross_profit
                ),

            "gross_loss":
                float(
                    gross_loss
                ),

            "best_trade":
                best_trade,

            "worst_trade":
                worst_trade,

            "profit_factor":
                profit_factor,

            "average_holding_days":
                average_holding_days,

            # ================================================
            # Risk
            # ================================================

            "max_drawdown":
                float(
                    max_drawdown
                ),

            "skipped_entries":
                int(
                    len(
                        self.skipped_entries
                    )
                ),

            # ================================================
            # Costs
            # ================================================

            "total_commission":
                float(
                    total_commission
                ),

            "total_slippage":
                float(
                    total_slippage
                ),

            "total_trading_cost":
                float(
                    total_trading_cost
                ),

            # ================================================
            # Accounting audit
            # ================================================

            "trade_profit_sum":
                float(
                    trade_profit_sum
                ),

            "accounting_difference":
                float(
                    accounting_difference
                ),

            "trade_audit_difference":
                float(
                    trade_audit_difference
                ),

            "accounting_ok":
                bool(
                    accounting_ok
                ),
        }


    # ========================================================
    # Getter
    # ========================================================

    def get_trades(
        self,
    ):

        return self.trades.copy()


    def get_equity_curve(
        self,
    ):

        return self.equity_curve.copy()


    def get_skipped_entries(
        self,
    ):

        return self.skipped_entries.copy()


    def get_order_log(
        self,
    ):

        return self.order_log.copy()


    def get_metrics(
        self,
    ):

        return dict(
            self.metrics
        )


    # ========================================================
    # Engine info
    # ========================================================

    def get_engine_info(
        self,
    ):

        return {

            "engine":
                "TradingBacktestEngine",

            "version":
                "v5.1",

            "execution_rule":
                (
                    "Signal at close / "
                    "Fill at next open"
                ),

            "same_day_exit_after_buy":
                False,

            "final_position":
                "Forced exit at final close",

            "initial_capital":
                self.initial_capital,

            "lot_size":
                self.lot_size,

            "risk_per_trade":
                self.risk_per_trade,

            "max_position_rate":
                self.max_position_rate,

            "commission_rate":
                self.commission_rate,

            "slippage_rate":
                self.slippage_rate,
        }
