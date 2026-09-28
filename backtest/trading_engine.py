# ============================================================
# backtest/trading_engine.py
#
# TradingBacktestEngine v5
#
# 売買タイミング固定
# 会計監査対応
# 保有日数修正版
# ============================================================

import numpy as np
import pandas as pd

from strategy.entry import EntryStrategy
from strategy.exit import ExitStrategy
from strategy.risk import RiskManager


class TradingBacktestEngine:

    def __init__(
        self,
        initial_capital=1_000_000,
        lot_size=100,

        entry_minimum_score=6,
        entry_minimum_probability=0.55,
        entry_strong_probability=0.60,

        stop_loss_rate=0.05,
        take_profit_rate=0.10,

        ai_exit_probability=0.45,
        max_holding_days=10,
        minimum_exit_score=3,

        risk_per_trade=0.01,
        max_position_rate=0.50,

        commission_rate=0.001,
        slippage_rate=0.001,
    ):

        self.initial_capital = float(
            initial_capital
        )

        self.lot_size = int(
            lot_size
        )

        self.commission_rate = float(
            commission_rate
        )

        self.slippage_rate = float(
            slippage_rate
        )

        self.entry_strategy = EntryStrategy(
            minimum_score=
                entry_minimum_score,

            minimum_probability=
                entry_minimum_probability,

            strong_probability=
                entry_strong_probability,
        )

        self.exit_strategy = ExitStrategy(
            stop_loss_rate=
                stop_loss_rate,

            take_profit_rate=
                take_profit_rate,

            ai_exit_probability=
                ai_exit_probability,

            max_holding_days=
                max_holding_days,

            minimum_exit_score=
                minimum_exit_score,
        )

        self.risk_manager = RiskManager(
            lot_size=
                lot_size,

            risk_per_trade=
                risk_per_trade,

            max_position_rate=
                max_position_rate,

            stop_loss_rate=
                stop_loss_rate,

            take_profit_rate=
                take_profit_rate,

            commission_rate=
                commission_rate,

            slippage_rate=
                slippage_rate,
        )

        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.metrics = {}

        self.skipped_entries = (
            pd.DataFrame()
        )

        self.order_log = (
            pd.DataFrame()
        )


    # ========================================================
    # 安全なfloat
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

                value = pd.to_numeric(
                    value,
                    errors="coerce",
                ).dropna()

                if value.empty:
                    return default

                value = value.iloc[-1]

            if isinstance(
                value,
                (
                    np.ndarray,
                    list,
                    tuple,
                ),
            ):

                value = np.asarray(
                    value
                ).reshape(-1)

                if len(value) == 0:
                    return default

                value = value[-1]

            number = float(value)

            if not np.isfinite(
                number
            ):

                return default

            return number

        except Exception:

            return default


    # ========================================================
    # index整理
    # ========================================================

    @staticmethod
    def _prepare_index(
        data,
    ):

        result = data.copy()

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

        return result.sort_index()


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        stock_data,
        ai_data,
        walk_results,
    ):

        if (
            stock_data is None
            or stock_data.empty
        ):

            raise ValueError(
                "stock_data がありません。"
            )

        if (
            ai_data is None
            or ai_data.empty
        ):

            raise ValueError(
                "ai_data がありません。"
            )

        if (
            walk_results is None
            or walk_results.empty
        ):

            raise ValueError(
                "walk_results がありません。"
            )

        stock = self._prepare_index(
            stock_data
        )

        features = self._prepare_index(
            ai_data
        )

        predictions = self._prepare_index(
            walk_results
        )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
        ]

        missing = [
            column
            for column in required
            if column
            not in stock.columns
        ]

        if missing:

            raise ValueError(
                "stock_data に必要な列がありません: "
                + ", ".join(
                    missing
                )
            )

        if (
            "Probability_Up"
            not in predictions.columns
        ):

            raise ValueError(
                "walk_results に "
                "Probability_Up がありません。"
            )

        for column in required:

            stock[column] = (
                pd.to_numeric(
                    stock[column],
                    errors="coerce",
                )
            )

        predictions[
            "Probability_Up"
        ] = pd.to_numeric(
            predictions[
                "Probability_Up"
            ],
            errors="coerce",
        )

        stock = stock.dropna(
            subset=[
                "Open",
                "Close",
            ]
        )

        predictions = (
            predictions.dropna(
                subset=[
                    "Probability_Up"
                ]
            )
        )

        return (
            stock,
            features,
            predictions,
        )


    # ========================================================
    # 次営業日
    # ========================================================

    @staticmethod
    def get_next_trading_date(
        stock_index,
        current_date,
    ):

        future = stock_index[
            stock_index > current_date
        ]

        if len(future) == 0:
            return None

        return future[0]


    # ========================================================
    # AI確率
    # ========================================================

    def get_probability(
        self,
        predictions,
        date,
    ):

        if date not in predictions.index:
            return None

        return self._safe_float(
            predictions.loc[
                date,
                "Probability_Up",
            ]
        )


    # ========================================================
    # 特徴量
    # ========================================================

    @staticmethod
    def get_feature_row(
        features,
        date,
    ):

        if date not in features.index:
            return None

        row = features.loc[
            date
        ]

        if isinstance(
            row,
            pd.DataFrame,
        ):

            row = row.iloc[-1]

        return row


    # ========================================================
    # 約定価格
    # ========================================================

    def calculate_buy_price(
        self,
        market_price,
    ):

        return (
            float(market_price)
            * (
                1.0
                + self.slippage_rate
            )
        )


    def calculate_sell_price(
        self,
        market_price,
    ):

        return (
            float(market_price)
            * (
                1.0
                - self.slippage_rate
            )
        )


    # ========================================================
    # 手数料
    # ========================================================

    def calculate_commission(
        self,
        price,
        shares,
    ):

        return (
            float(price)
            * int(shares)
            * self.commission_rate
        )


    # ========================================================
    # スリッページ金額
    # ========================================================

    @staticmethod
    def calculate_buy_slippage(
        market_price,
        execution_price,
        shares,
    ):

        return max(
            0.0,
            (
                float(execution_price)
                - float(market_price)
            )
            * int(shares),
        )


    @staticmethod
    def calculate_sell_slippage(
        market_price,
        execution_price,
        shares,
    ):

        return max(
            0.0,
            (
                float(market_price)
                - float(execution_price)
            )
            * int(shares),
        )


    # ========================================================
    # 保有営業日数
    #
    # BUY日 = 0
    # 翌営業日 = 1
    # 2営業日後 = 2
    # ========================================================

    @staticmethod
    def calculate_holding_days(
        stock_index,
        entry_date,
        current_date,
    ):

        try:

            entry_location = (
                stock_index.get_loc(
                    entry_date
                )
            )

            current_location = (
                stock_index.get_loc(
                    current_date
                )
            )

            if isinstance(
                entry_location,
                slice,
            ):

                entry_location = (
                    entry_location.start
                )

            if isinstance(
                current_location,
                slice,
            ):

                current_location = (
                    current_location.start
                )

            days = (
                int(current_location)
                - int(entry_location)
            )

            return max(
                0,
                days,
            )

        except Exception:

            return 0


    # ========================================================
    # Exit判定
    # ========================================================

    def evaluate_exit_safely(
        self,
        row,
        probability,
        entry_price,
        holding_days,
    ):

        current_price = (
            self._safe_float(
                row.get(
                    "Close"
                )
            )
        )

        if current_price is None:

            return {
                "action":
                    "HOLD",

                "reason":
                    "NO_CLOSE",
            }

        if probability is not None:

            try:

                result = (
                    self.exit_strategy
                    .evaluate(
                        row=row,
                        probability=
                            probability,
                        entry_price=
                            entry_price,
                        holding_days=
                            holding_days,
                    )
                )

                if isinstance(
                    result,
                    dict,
                ):

                    return result

            except TypeError:

                try:

                    result = (
                        self.exit_strategy
                        .evaluate(
                            row,
                            probability,
                            entry_price,
                            holding_days,
                        )
                    )

                    if isinstance(
                        result,
                        dict,
                    ):

                        return result

                except Exception:
                    pass

            except Exception:
                pass

        # ====================================================
        # AI確率がない場合も
        # ハードExitだけ判定
        # ====================================================

        return_rate = (
            current_price
            / entry_price
            - 1.0
        )

        stop_loss = float(
            getattr(
                self.exit_strategy,
                "stop_loss_rate",
                0.05,
            )
        )

        take_profit = float(
            getattr(
                self.exit_strategy,
                "take_profit_rate",
                0.10,
            )
        )

        max_days = int(
            getattr(
                self.exit_strategy,
                "max_holding_days",
                10,
            )
        )

        if return_rate <= -stop_loss:

            return {
                "action":
                    "SELL",

                "reason":
                    "STOP_LOSS",
            }

        if return_rate >= take_profit:

            return {
                "action":
                    "SELL",

                "reason":
                    "TAKE_PROFIT",
            }

        if holding_days >= max_days:

            return {
                "action":
                    "SELL",

                "reason":
                    "MAX_HOLDING_DAYS",
            }

        return {
            "action":
                "HOLD",

            "reason":
                "HOLD",
        }


    # ========================================================
    # 取引レコード
    # ========================================================

    def create_trade_record(
        self,
        position,
        exit_signal_date,
        exit_date,
        exit_market_price,
        exit_price,
        exit_probability,
        exit_reason,
        forced_exit,
        cash_after_exit,
        holding_days,
    ):

        shares = int(
            position[
                "shares"
            ]
        )

        entry_market_price = float(
            position[
                "entry_market_price"
            ]
        )

        entry_price = float(
            position[
                "entry_price"
            ]
        )

        entry_value = float(
            position[
                "entry_value"
            ]
        )

        buy_commission = float(
            position[
                "buy_commission"
            ]
        )

        buy_slippage = float(
            position[
                "buy_slippage"
            ]
        )

        exit_market_price = float(
            exit_market_price
        )

        exit_price = float(
            exit_price
        )

        exit_value = (
            exit_price
            * shares
        )

        sell_commission = (
            self.calculate_commission(
                exit_price,
                shares,
            )
        )

        sell_slippage = (
            self.calculate_sell_slippage(
                exit_market_price,
                exit_price,
                shares,
            )
        )

        total_entry_cost = (
            entry_value
            + buy_commission
        )

        net_exit_proceeds = (
            exit_value
            - sell_commission
        )

        net_profit = (
            net_exit_proceeds
            - total_entry_cost
        )

        if total_entry_cost > 0:

            net_return = (
                net_profit
                / total_entry_cost
            )

        else:

            net_return = 0.0

        total_commission = (
            buy_commission
            + sell_commission
        )

        total_slippage = (
            buy_slippage
            + sell_slippage
        )

        total_trading_cost = (
            total_commission
            + total_slippage
        )

        # ====================================================
        # スリッページ・手数料なしの
        # 市場価格ベース損益
        # ====================================================

        gross_market_profit = (
            (
                exit_market_price
                - entry_market_price
            )
            * shares
        )

        # ====================================================
        # 別経路で純損益を再計算
        # ====================================================

        audit_net_profit = (
            gross_market_profit
            - total_slippage
            - total_commission
        )

        trade_audit_difference = (
            net_profit
            - audit_net_profit
        )

        trade_audit_ok = (
            abs(
                trade_audit_difference
            )
            < 0.01
        )

        return {

            "Entry_Signal_Date":
                position[
                    "signal_date"
                ],

            "Entry_Date":
                position[
                    "entry_date"
                ],

            "Exit_Signal_Date":
                exit_signal_date,

            "Exit_Date":
                exit_date,

            "Shares":
                shares,

            "Entry_Market_Price":
                entry_market_price,

            "Entry_Price":
                entry_price,

            "Entry_Value":
                entry_value,

            "Buy_Commission":
                buy_commission,

            "Buy_Slippage":
                buy_slippage,

            "Exit_Market_Price":
                exit_market_price,

            "Exit_Price":
                exit_price,

            "Exit_Value":
                exit_value,

            "Sell_Commission":
                sell_commission,

            "Sell_Slippage":
                sell_slippage,

            "Total_Commission":
                total_commission,

            "Total_Slippage":
                total_slippage,

            "Total_Trading_Cost":
                total_trading_cost,

            "Gross_Market_Profit":
                gross_market_profit,

            "Net_Profit":
                net_profit,

            "Net_Return":
                net_return,

            "Audit_Net_Profit":
                audit_net_profit,

            "Trade_Audit_Difference":
                trade_audit_difference,

            "Trade_Audit_OK":
                trade_audit_ok,

            "Holding_Days":
                int(
                    holding_days
                ),

            "Entry_Probability":
                position[
                    "entry_probability"
                ],

            "Exit_Probability":
                exit_probability,

            "Entry_Score":
                position[
                    "entry_score"
                ],

            "Exit_Reason":
                exit_reason,

            "Forced_Exit":
                forced_exit,

            "Cash_After_Exit":
                cash_after_exit,
        }


    # ========================================================
    # メイン
    # ========================================================

    def run(
        self,
        stock_data,
        ai_data,
        walk_results,
    ):

        (
            stock,
            features,
            predictions,
        ) = self.prepare_data(
            stock_data,
            ai_data,
            walk_results,
        )

        cash = float(
            self.initial_capital
        )

        position = None

        pending_order = None

        trades = []

        equity_records = []

        skipped_entries = []

        order_records = []

        prediction_start = (
            predictions.index.min()
        )

        trading_dates = stock.index[
            stock.index
            >= prediction_start
        ]

        if len(trading_dates) == 0:

            raise ValueError(
                "バックテスト可能な"
                "日付がありません。"
            )


        # ====================================================
        # 日次ループ
        # ====================================================

        for current_date in trading_dates:

            stock_row = stock.loc[
                current_date
            ]

            if isinstance(
                stock_row,
                pd.DataFrame,
            ):

                stock_row = (
                    stock_row.iloc[-1]
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
                or market_close is None
            ):

                continue


            # =================================================
            # 今日BUYしたか
            #
            # これがTrueなら
            # 今日の終値ではSELL判定しない。
            # =================================================

            bought_today = False


            # =================================================
            # A. 始値で注文執行
            # =================================================

            if pending_order is not None:

                execution_date = (
                    pending_order.get(
                        "execution_date"
                    )
                )

                if (
                    execution_date is None
                    or current_date
                    >= execution_date
                ):

                    order_type = (
                        pending_order.get(
                            "type"
                        )
                    )


                    # =========================================
                    # BUY
                    # =========================================

                    if (
                        order_type == "BUY"
                        and position is None
                    ):

                        signal_date = (
                            pending_order.get(
                                "signal_date"
                            )
                        )

                        signal_probability = (
                            pending_order.get(
                                "probability"
                            )
                        )

                        signal_score = (
                            pending_order.get(
                                "score"
                            )
                        )

                        risk_result = (
                            self.risk_manager
                            .evaluate_trade(
                                capital=cash,
                                market_price=
                                    market_open,
                            )
                        )

                        can_trade = bool(
                            risk_result.get(
                                "can_trade",
                                False,
                            )
                        )

                        shares = int(
                            self._safe_float(
                                risk_result.get(
                                    "shares"
                                ),
                                0,
                            )
                        )

                        shares = (
                            shares
                            // self.lot_size
                            * self.lot_size
                        )

                        buy_price = (
                            self.calculate_buy_price(
                                market_open
                            )
                        )

                        entry_value = (
                            buy_price
                            * shares
                        )

                        buy_commission = (
                            self.calculate_commission(
                                buy_price,
                                shares,
                            )
                        )

                        buy_slippage = (
                            self.calculate_buy_slippage(
                                market_open,
                                buy_price,
                                shares,
                            )
                        )

                        required_cash = (
                            entry_value
                            + buy_commission
                        )

                        if (
                            shares <= 0
                            or required_cash
                            > cash + 1e-9
                        ):

                            can_trade = False


                        if can_trade:

                            cash -= (
                                required_cash
                            )

                            position = {

                                "signal_date":
                                    signal_date,

                                "entry_date":
                                    current_date,

                                "shares":
                                    shares,

                                "entry_market_price":
                                    market_open,

                                "entry_price":
                                    buy_price,

                                "entry_value":
                                    entry_value,

                                "buy_commission":
                                    buy_commission,

                                "buy_slippage":
                                    buy_slippage,

                                "entry_probability":
                                    signal_probability,

                                "entry_score":
                                    signal_score,
                            }

                            bought_today = True

                            order_records.append({

                                "Date":
                                    current_date,

                                "Signal_Date":
                                    signal_date,

                                "Event":
                                    "BUY_FILLED",

                                "Market_Price":
                                    market_open,

                                "Execution_Price":
                                    buy_price,

                                "Shares":
                                    shares,

                                "Trade_Value":
                                    entry_value,

                                "Commission":
                                    buy_commission,

                                "Slippage":
                                    buy_slippage,

                                "Cash_After":
                                    cash,
                            })

                        else:

                            reason = (
                                risk_result.get(
                                    "reason",
                                    "資金管理条件で見送り",
                                )
                            )

                            skipped_entries.append({

                                "Date":
                                    current_date,

                                "Signal_Date":
                                    signal_date,

                                "Market_Price":
                                    market_open,

                                "Calculated_Buy_Price":
                                    buy_price,

                                "Shares":
                                    shares,

                                "Required_Cash":
                                    required_cash,

                                "Available_Cash":
                                    cash,

                                "Probability_Up":
                                    signal_probability,

                                "Score":
                                    signal_score,

                                "Reason":
                                    reason,
                            })

                            order_records.append({

                                "Date":
                                    current_date,

                                "Signal_Date":
                                    signal_date,

                                "Event":
                                    "BUY_SKIPPED",

                                "Market_Price":
                                    market_open,

                                "Shares":
                                    shares,

                                "Required_Cash":
                                    required_cash,

                                "Cash":
                                    cash,

                                "Reason":
                                    reason,
                            })

                        pending_order = None


                    # =========================================
                    # SELL
                    # =========================================

                    elif (
                        order_type == "SELL"
                        and position is not None
                    ):

                        signal_date = (
                            pending_order.get(
                                "signal_date"
                            )
                        )

                        exit_reason = (
                            pending_order.get(
                                "reason",
                                "SELL",
                            )
                        )

                        exit_probability = (
                            pending_order.get(
                                "probability"
                            )
                        )

                        shares = int(
                            position[
                                "shares"
                            ]
                        )

                        sell_price = (
                            self.calculate_sell_price(
                                market_open
                            )
                        )

                        sell_value = (
                            sell_price
                            * shares
                        )

                        sell_commission = (
                            self.calculate_commission(
                                sell_price,
                                shares,
                            )
                        )

                        net_sell_proceeds = (
                            sell_value
                            - sell_commission
                        )

                        cash += (
                            net_sell_proceeds
                        )

                        holding_days = (
                            self.calculate_holding_days(
                                stock.index,
                                position[
                                    "entry_date"
                                ],
                                current_date,
                            )
                        )

                        trade_record = (
                            self.create_trade_record(
                                position=
                                    position,

                                exit_signal_date=
                                    signal_date,

                                exit_date=
                                    current_date,

                                exit_market_price=
                                    market_open,

                                exit_price=
                                    sell_price,

                                exit_probability=
                                    exit_probability,

                                exit_reason=
                                    exit_reason,

                                forced_exit=
                                    False,

                                cash_after_exit=
                                    cash,

                                holding_days=
                                    holding_days,
                            )
                        )

                        trades.append(
                            trade_record
                        )

                        order_records.append({

                            "Date":
                                current_date,

                            "Signal_Date":
                                signal_date,

                            "Event":
                                "SELL_FILLED",

                            "Market_Price":
                                market_open,

                            "Execution_Price":
                                sell_price,

                            "Shares":
                                shares,

                            "Commission":
                                trade_record[
                                    "Sell_Commission"
                                ],

                            "Slippage":
                                trade_record[
                                    "Sell_Slippage"
                                ],

                            "Net_Profit":
                                trade_record[
                                    "Net_Profit"
                                ],

                            "Cash_After":
                                cash,

                            "Reason":
                                exit_reason,
                        })

                        position = None

                        pending_order = None


            # =================================================
            # B. 当日終値情報
            # =================================================

            probability = (
                self.get_probability(
                    predictions,
                    current_date,
                )
            )

            feature_row = (
                self.get_feature_row(
                    features,
                    current_date,
                )
            )


            # =================================================
            # C. SELL判定
            #
            # 重要:
            # BUY約定当日はSELL判定しない。
            # =================================================

            if (
                position is not None
                and pending_order is None
                and not bought_today
            ):

                holding_days = (
                    self.calculate_holding_days(
                        stock.index,
                        position[
                            "entry_date"
                        ],
                        current_date,
                    )
                )

                # ---------------------------------------------
                # 最低1営業日保有後から判定
                # ---------------------------------------------

                if (
                    holding_days >= 1
                    and feature_row is not None
                ):

                    exit_result = (
                        self.evaluate_exit_safely(
                            row=
                                feature_row,

                            probability=
                                probability,

                            entry_price=
                                position[
                                    "entry_price"
                                ],

                            holding_days=
                                holding_days,
                        )
                    )

                    if (
                        exit_result.get(
                            "action"
                        )
                        == "SELL"
                    ):

                        next_date = (
                            self.get_next_trading_date(
                                stock.index,
                                current_date,
                            )
                        )

                        if next_date is not None:

                            pending_order = {

                                "type":
                                    "SELL",

                                "signal_date":
                                    current_date,

                                "execution_date":
                                    next_date,

                                "reason":
                                    exit_result.get(
                                        "reason",
                                        exit_result.get(
                                            "exit_reason",
                                            "SELL",
                                        ),
                                    ),

                                "probability":
                                    probability,
                            }

                            order_records.append({

                                "Date":
                                    current_date,

                                "Event":
                                    "SELL_SIGNAL",

                                "Execution_Date":
                                    next_date,

                                "Close":
                                    market_close,

                                "Holding_Days":
                                    holding_days,

                                "Probability_Up":
                                    probability,

                                "Reason":
                                    pending_order[
                                        "reason"
                                    ],
                            })


            # =================================================
            # D. BUY判定
            # =================================================

            if (
                position is None
                and pending_order is None
                and probability is not None
                and feature_row is not None
            ):

                try:

                    entry_result = (
                        self.entry_strategy
                        .evaluate(
                            row=
                                feature_row,

                            probability=
                                probability,
                        )
                    )

                except TypeError:

                    entry_result = (
                        self.entry_strategy
                        .evaluate(
                            feature_row,
                            probability,
                        )
                    )

                if (
                    entry_result.get(
                        "action"
                    )
                    == "BUY"
                ):

                    next_date = (
                        self.get_next_trading_date(
                            stock.index,
                            current_date,
                        )
                    )

                    if next_date is not None:

                        pending_order = {

                            "type":
                                "BUY",

                            "signal_date":
                                current_date,

                            "execution_date":
                                next_date,

                            "probability":
                                probability,

                            "score":
                                entry_result.get(
                                    "score"
                                ),
                        }

                        order_records.append({

                            "Date":
                                current_date,

                            "Event":
                                "BUY_SIGNAL",

                            "Execution_Date":
                                next_date,

                            "Close":
                                market_close,

                            "Probability_Up":
                                probability,

                            "Score":
                                entry_result.get(
                                    "score"
                                ),
                        })


            # =================================================
            # E. 日次資産
            # =================================================

            if position is not None:

                position_value = (
                    market_close
                    * position[
                        "shares"
                    ]
                )

            else:

                position_value = 0.0

            total_equity = (
                cash
                + position_value
            )

            equity_records.append({

                "Date":
                    current_date,

                "Cash":
                    cash,

                "Position_Value":
                    position_value,

                "Total_Equity":
                    total_equity,

                "Shares":
                    (
                        position[
                            "shares"
                        ]
                        if position is not None
                        else 0
                    ),

                "Close":
                    market_close,
            })


        # ====================================================
        # F. 最終日強制決済
        # ====================================================

        if position is not None:

            final_date = (
                trading_dates[-1]
            )

            final_row = stock.loc[
                final_date
            ]

            if isinstance(
                final_row,
                pd.DataFrame,
            ):

                final_row = (
                    final_row.iloc[-1]
                )

            final_close = (
                self._safe_float(
                    final_row.get(
                        "Close"
                    )
                )
            )

            if final_close is not None:

                shares = int(
                    position[
                        "shares"
                    ]
                )

                sell_price = (
                    self.calculate_sell_price(
                        final_close
                    )
                )

                sell_value = (
                    sell_price
                    * shares
                )

                sell_commission = (
                    self.calculate_commission(
                        sell_price,
                        shares,
                    )
                )

                cash += (
                    sell_value
                    - sell_commission
                )

                holding_days = (
                    self.calculate_holding_days(
                        stock.index,
                        position[
                            "entry_date"
                        ],
                        final_date,
                    )
                )

                trade_record = (
                    self.create_trade_record(
                        position=
                            position,

                        exit_signal_date=
                            final_date,

                        exit_date=
                            final_date,

                        exit_market_price=
                            final_close,

                        exit_price=
                            sell_price,

                        exit_probability=
                            None,

                        exit_reason=
                            "FINAL_DATA_EXIT",

                        forced_exit=
                            True,

                        cash_after_exit=
                            cash,

                        holding_days=
                            holding_days,
                    )
                )

                trades.append(
                    trade_record
                )

                order_records.append({

                    "Date":
                        final_date,

                    "Event":
                        "FINAL_SELL",

                    "Market_Price":
                        final_close,

                    "Execution_Price":
                        sell_price,

                    "Shares":
                        shares,

                    "Commission":
                        trade_record[
                            "Sell_Commission"
                        ],

                    "Slippage":
                        trade_record[
                            "Sell_Slippage"
                        ],

                    "Net_Profit":
                        trade_record[
                            "Net_Profit"
                        ],

                    "Cash_After":
                        cash,

                    "Reason":
                        "FINAL_DATA_EXIT",
                })

                position = None

                if equity_records:

                    equity_records[-1][
                        "Cash"
                    ] = cash

                    equity_records[-1][
                        "Position_Value"
                    ] = 0.0

                    equity_records[-1][
                        "Total_Equity"
                    ] = cash

                    equity_records[-1][
                        "Shares"
                    ] = 0


        # ====================================================
        # DataFrame化
        # ====================================================

        self.trades = pd.DataFrame(
            trades
        )

        self.equity_curve = (
            pd.DataFrame(
                equity_records
            )
        )

        if not self.equity_curve.empty:

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
                .sort_values(
                    "Date"
                )
                .set_index(
                    "Date"
                )
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

        self.metrics = (
            self.calculate_metrics(
                final_cash=cash
            )
        )

        return (
            self.trades,
            self.equity_curve,
            self.metrics,
        )


    # ========================================================
    # 最大DD
    # ========================================================

    def _calculate_max_drawdown(
        self,
    ):

        if (
            self.equity_curve.empty
            or "Total_Equity"
            not in self.equity_curve.columns
        ):

            return 0.0

        equity = pd.to_numeric(
            self.equity_curve[
                "Total_Equity"
            ],
            errors="coerce",
        ).dropna()

        if equity.empty:
            return 0.0

        running_max = (
            equity.cummax()
        )

        valid = (
            running_max > 0
        )

        equity = equity[
            valid
        ]

        running_max = (
            running_max[
                valid
            ]
        )

        if equity.empty:
            return 0.0

        drawdown = (
            equity
            / running_max
            - 1.0
        )

        return abs(
            float(
                drawdown.min()
            )
        )


    # ========================================================
    # 成績・会計監査
    # ========================================================

    def calculate_metrics(
        self,
        final_cash,
    ):

        final_capital = float(
            final_cash
        )

        asset_profit = (
            final_capital
            - self.initial_capital
        )

        total_return = (
            asset_profit
            / self.initial_capital
            if self.initial_capital > 0
            else 0.0
        )


        # ====================================================
        # 取引なし
        # ====================================================

        if self.trades.empty:

            accounting_ok = (
                abs(
                    asset_profit
                )
                < 0.01
            )

            return {

                "initial_capital":
                    self.initial_capital,

                "final_capital":
                    final_capital,

                "total_profit":
                    asset_profit,

                "total_return":
                    total_return,

                "trade_count":
                    0,

                "trades":
                    0,

                "wins":
                    0,

                "losses":
                    0,

                "flat":
                    0,

                "win_rate":
                    None,

                "average_profit":
                    None,

                "average_profit_rate":
                    None,

                "gross_profit":
                    0.0,

                "gross_loss":
                    0.0,

                "profit_factor":
                    None,

                "max_drawdown":
                    self._calculate_max_drawdown(),

                "average_holding_days":
                    None,

                "best_trade":
                    None,

                "worst_trade":
                    None,

                "buy_commission":
                    0.0,

                "sell_commission":
                    0.0,

                "total_commission":
                    0.0,

                "buy_slippage":
                    0.0,

                "sell_slippage":
                    0.0,

                "total_slippage":
                    0.0,

                "total_trading_cost":
                    0.0,

                "trade_profit_sum":
                    0.0,

                "win_loss_profit":
                    0.0,

                "asset_profit":
                    asset_profit,

                "accounting_difference":
                    asset_profit,

                "win_loss_difference":
                    0.0,

                "trade_audit_max_difference":
                    0.0,

                "all_trade_audit_ok":
                    True,

                "accounting_ok":
                    accounting_ok,

                "skipped_entries":
                    len(
                        self.skipped_entries
                    ),

                "order_signals":
                    len(
                        self.order_log
                    ),
            }


        # ====================================================
        # 損益
        # ====================================================

        profits = pd.to_numeric(
            self.trades[
                "Net_Profit"
            ],
            errors="coerce",
        ).fillna(0.0)

        returns = pd.to_numeric(
            self.trades[
                "Net_Return"
            ],
            errors="coerce",
        ).fillna(0.0)

        trade_count = int(
            len(profits)
        )

        wins = int(
            (
                profits > 0
            ).sum()
        )

        losses = int(
            (
                profits < 0
            ).sum()
        )

        flat = int(
            (
                profits == 0
            ).sum()
        )

        win_rate = (
            wins
            / trade_count
            if trade_count > 0
            else None
        )

        trade_profit_sum = float(
            profits.sum()
        )

        gross_profit = float(
            profits[
                profits > 0
            ].sum()
        )

        gross_loss = abs(
            float(
                profits[
                    profits < 0
                ].sum()
            )
        )

        win_loss_profit = (
            gross_profit
            - gross_loss
        )


        # ====================================================
        # PF
        # ====================================================

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        elif gross_profit > 0:

            profit_factor = (
                float("inf")
            )

        else:

            profit_factor = None


        # ====================================================
        # 手数料
        # ====================================================

        buy_commission = float(
            pd.to_numeric(
                self.trades[
                    "Buy_Commission"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        sell_commission = float(
            pd.to_numeric(
                self.trades[
                    "Sell_Commission"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        total_commission = (
            buy_commission
            + sell_commission
        )


        # ====================================================
        # スリッページ
        # ====================================================

        buy_slippage = float(
            pd.to_numeric(
                self.trades[
                    "Buy_Slippage"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        sell_slippage = float(
            pd.to_numeric(
                self.trades[
                    "Sell_Slippage"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        total_slippage = (
            buy_slippage
            + sell_slippage
        )

        total_trading_cost = (
            total_commission
            + total_slippage
        )


        # ====================================================
        # 保有日数
        # ====================================================

        holding = pd.to_numeric(
            self.trades[
                "Holding_Days"
            ],
            errors="coerce",
        )

        average_holding_days = (
            float(
                holding.mean()
            )
            if holding.notna().any()
            else None
        )


        # ====================================================
        # 会計監査
        # ====================================================

        accounting_difference = (
            asset_profit
            - trade_profit_sum
        )

        win_loss_difference = (
            trade_profit_sum
            - win_loss_profit
        )

        audit_difference = (
            pd.to_numeric(
                self.trades[
                    "Trade_Audit_Difference"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .abs()
        )

        if audit_difference.empty:

            trade_audit_max_difference = (
                0.0
            )

            all_trade_audit_ok = True

        else:

            trade_audit_max_difference = (
                float(
                    audit_difference.max()
                )
            )

            all_trade_audit_ok = bool(
                (
                    audit_difference
                    < 0.01
                ).all()
            )

        accounting_ok = bool(

            abs(
                accounting_difference
            ) < 0.01

            and

            abs(
                win_loss_difference
            ) < 0.01

            and

            all_trade_audit_ok
        )


        return {

            "initial_capital":
                self.initial_capital,

            "final_capital":
                final_capital,

            "total_profit":
                asset_profit,

            "total_return":
                total_return,

            "trade_count":
                trade_count,

            "trades":
                trade_count,

            "wins":
                wins,

            "losses":
                losses,

            "flat":
                flat,

            "win_rate":
                win_rate,

            "average_profit":
                float(
                    profits.mean()
                ),

            "average_profit_rate":
                float(
                    returns.mean()
                ),

            "gross_profit":
                gross_profit,

            "gross_loss":
                gross_loss,

            "profit_factor":
                profit_factor,

            "max_drawdown":
                self._calculate_max_drawdown(),

            "average_holding_days":
                average_holding_days,

            "best_trade":
                float(
                    profits.max()
                ),

            "worst_trade":
                float(
                    profits.min()
                ),

            # ================================================
            # 手数料
            # ================================================

            "buy_commission":
                buy_commission,

            "sell_commission":
                sell_commission,

            "total_commission":
                total_commission,

            # ================================================
            # スリッページ
            # ================================================

            "buy_slippage":
                buy_slippage,

            "sell_slippage":
                sell_slippage,

            "total_slippage":
                total_slippage,

            "total_trading_cost":
                total_trading_cost,

            # ================================================
            # 会計
            # ================================================

            "trade_profit_sum":
                trade_profit_sum,

            "win_loss_profit":
                win_loss_profit,

            "asset_profit":
                asset_profit,

            "accounting_difference":
                accounting_difference,

            "win_loss_difference":
                win_loss_difference,

            "trade_audit_max_difference":
                trade_audit_max_difference,

            "all_trade_audit_ok":
                all_trade_audit_ok,

            "accounting_ok":
                accounting_ok,

            "skipped_entries":
                len(
                    self.skipped_entries
                ),

            "order_signals":
                len(
                    self.order_log
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

        return (
            self.equity_curve.copy()
        )


    def get_metrics(
        self,
    ):

        return dict(
            self.metrics
        )


    def get_skipped_entries(
        self,
    ):

        return (
            self.skipped_entries.copy()
        )


    def get_order_log(
        self,
    ):

        return (
            self.order_log.copy()
        )


# ============================================================
# 簡易実行
# ============================================================

def run_trading_backtest(
    stock_data,
    ai_data,
    walk_results,
    initial_capital=1_000_000,
    lot_size=100,
):

    engine = TradingBacktestEngine(
        initial_capital=
            initial_capital,

        lot_size=
            lot_size,
    )

    return engine.run(
        stock_data=
            stock_data,

        ai_data=
            ai_data,

        walk_results=
            walk_results,
    )
