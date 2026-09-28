# ============================================================
# backtest/trading_engine.py
#
# 本格戦略バックテストエンジン
# 会計監査対応 完全版 v4
#
# ------------------------------------------------------------
# 約定ルール
# ・当日終値でBUY / SELL判定
# ・翌営業日の始値で約定
#
# コスト
# ・BUY  : 始値 × (1 + スリッページ率)
# ・SELL : 始値 × (1 - スリッページ率)
# ・BUY手数料  = BUY約定金額 × 手数料率
# ・SELL手数料 = SELL約定金額 × 手数料率
#
# 純損益
# ・売却手取額
#   - 購入代金
#   - BUY手数料
#
# 会計監査
# ・全取引Net_Profit合計
# ・勝ち利益 - 負け損失
# ・最終資産 - 初期資金
# ・3経路を相互照合
# ・BUY/SELL手数料を個別集計
# ・BUY/SELLスリッページを個別集計
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

        # ====================================================
        # 基本設定
        # ====================================================

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


        # ====================================================
        # Entry
        # ====================================================

        self.entry_strategy = EntryStrategy(
            minimum_score=
                entry_minimum_score,

            minimum_probability=
                entry_minimum_probability,

            strong_probability=
                entry_strong_probability,
        )


        # ====================================================
        # Exit
        # ====================================================

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


        # ====================================================
        # Risk
        # ====================================================

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


        # ====================================================
        # 結果保存
        # ====================================================

        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.metrics = {}

        self.skipped_entries = pd.DataFrame()

        self.order_log = pd.DataFrame()


    # ========================================================
    # 安全なfloat変換
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

                numeric = pd.to_numeric(
                    value,
                    errors="coerce",
                ).dropna()

                if numeric.empty:
                    return default

                value = numeric.iloc[-1]

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


        required_stock = [
            "Open",
            "High",
            "Low",
            "Close",
        ]


        missing_stock = [

            column

            for column in required_stock

            if column not in stock.columns
        ]


        if missing_stock:

            raise ValueError(
                "stock_data に必要な列がありません: "
                + ", ".join(
                    missing_stock
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


        for column in required_stock:

            stock[column] = pd.to_numeric(
                stock[column],
                errors="coerce",
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
    # 次の営業日
    # ========================================================

    @staticmethod
    def get_next_trading_date(
        stock_index,
        current_date,
    ):

        future_dates = stock_index[
            stock_index > current_date
        ]

        if len(future_dates) == 0:
            return None

        return future_dates[0]


    # ========================================================
    # BUY約定価格
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


    # ========================================================
    # SELL約定価格
    # ========================================================

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
        execution_price,
        shares,
    ):

        return (
            float(
                execution_price
            )
            * int(
                shares
            )
            * self.commission_rate
        )


    # ========================================================
    # BUYスリッページ
    # ========================================================

    @staticmethod
    def calculate_buy_slippage_cost(
        market_price,
        execution_price,
        shares,
    ):

        return max(
            0.0,
            (
                float(
                    execution_price
                )
                - float(
                    market_price
                )
            )
            * int(
                shares
            ),
        )


    # ========================================================
    # SELLスリッページ
    # ========================================================

    @staticmethod
    def calculate_sell_slippage_cost(
        market_price,
        execution_price,
        shares,
    ):

        return max(
            0.0,
            (
                float(
                    market_price
                )
                - float(
                    execution_price
                )
            )
            * int(
                shares
            ),
        )


    # ========================================================
    # Exit安全判定
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


        # ====================================================
        # 通常のExitStrategy
        # ====================================================

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
        # AI確率がない場合のハードExit
        # ====================================================

        return_rate = (
            current_price
            / entry_price
            - 1.0
        )


        stop_loss_rate = float(
            getattr(
                self.exit_strategy,
                "stop_loss_rate",
                0.05,
            )
        )


        take_profit_rate = float(
            getattr(
                self.exit_strategy,
                "take_profit_rate",
                0.10,
            )
        )


        max_holding_days = int(
            getattr(
                self.exit_strategy,
                "max_holding_days",
                10,
            )
        )


        if (
            return_rate
            <= -stop_loss_rate
        ):

            return {
                "action":
                    "SELL",

                "reason":
                    "STOP_LOSS",
            }


        if (
            return_rate
            >= take_profit_rate
        ):

            return {
                "action":
                    "SELL",

                "reason":
                    "TAKE_PROFIT",
            }


        if (
            holding_days
            >= max_holding_days
        ):

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
    # 取引レコード作成
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
    ):

        shares = int(
            position[
                "shares"
            ]
        )


        # ====================================================
        # SELL代金
        # ====================================================

        exit_value = (
            float(
                exit_price
            )
            * shares
        )


        # ====================================================
        # SELL手数料
        # ====================================================

        sell_commission = (
            self.calculate_commission(
                exit_price,
                shares,
            )
        )


        # ====================================================
        # SELLスリッページ
        # ====================================================

        sell_slippage = (
            self.calculate_sell_slippage_cost(
                exit_market_price,
                exit_price,
                shares,
            )
        )


        # ====================================================
        # BUY側
        # ====================================================

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


        # ====================================================
        # 実際にBUYで支払った総額
        # ====================================================

        total_entry_cost = (
            entry_value
            + buy_commission
        )


        # ====================================================
        # 実際にSELLで受け取る総額
        # ====================================================

        net_exit_proceeds = (
            exit_value
            - sell_commission
        )


        # ====================================================
        # 純損益
        #
        # SELL手取額
        # -
        # BUY総支払額
        # ====================================================

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


        # ====================================================
        # 手数料
        # ====================================================

        total_commission = (
            buy_commission
            + sell_commission
        )


        # ====================================================
        # スリッページ
        # ====================================================

        total_slippage = (
            buy_slippage
            + sell_slippage
        )


        # ====================================================
        # コストなし市場損益
        #
        # Entryの市場価格
        # →
        # Exitの市場価格
        # ====================================================

        gross_market_profit = (
            (
                float(
                    exit_market_price
                )
                - float(
                    position[
                        "entry_market_price"
                    ]
                )
            )
            * shares
        )


        # ====================================================
        # コスト総額
        #
        # 手数料 + スリッページ
        # ====================================================

        total_trading_cost = (
            total_commission
            + total_slippage
        )


        # ====================================================
        # 監査用理論純損益
        #
        # 市場価格ベース利益
        # -
        # スリッページ
        # -
        # 手数料
        #
        # 小数誤差を除きNet_Profitと一致するはず
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

            # ================================================
            # BUY
            # ================================================

            "Entry_Market_Price":
                position[
                    "entry_market_price"
                ],

            "Entry_Price":
                position[
                    "entry_price"
                ],

            "Entry_Value":
                entry_value,

            "Buy_Commission":
                buy_commission,

            "Buy_Slippage":
                buy_slippage,

            # ================================================
            # SELL
            # ================================================

            "Exit_Market_Price":
                float(
                    exit_market_price
                ),

            "Exit_Price":
                float(
                    exit_price
                ),

            "Exit_Value":
                exit_value,

            "Sell_Commission":
                sell_commission,

            "Sell_Slippage":
                sell_slippage,

            # ================================================
            # 合計
            # ================================================

            "Total_Commission":
                total_commission,

            "Total_Slippage":
                total_slippage,

            "Total_Trading_Cost":
                total_trading_cost,

            # ================================================
            # 損益
            # ================================================

            "Gross_Market_Profit":
                gross_market_profit,

            "Net_Profit":
                net_profit,

            "Net_Return":
                net_return,

            # ================================================
            # 監査
            # ================================================

            "Audit_Net_Profit":
                audit_net_profit,

            "Trade_Audit_Difference":
                trade_audit_difference,

            "Trade_Audit_OK":
                (
                    abs(
                        trade_audit_difference
                    )
                    < 0.01
                ),

            # ================================================
            # その他
            # ================================================

            "Holding_Days":
                position[
                    "holding_days"
                ],

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
    # メインバックテスト
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


        # ====================================================
        # 初期状態
        # ====================================================

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
        # 日次処理
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
            # A. 始値でPending注文を実行
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


                        probability = (
                            pending_order.get(
                                "probability"
                            )
                        )


                        score = (
                            pending_order.get(
                                "score"
                            )
                        )


                        # =====================================
                        # RiskManagerで株数を決定
                        # =====================================

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


                        # =====================================
                        # 売買単位へ丸める
                        # =====================================

                        shares = (
                            shares
                            // self.lot_size
                            * self.lot_size
                        )


                        # =====================================
                        # BUY約定価格
                        # =====================================

                        buy_price = (
                            self.calculate_buy_price(
                                market_open
                            )
                        )


                        # =====================================
                        # BUY約定金額
                        # =====================================

                        entry_value = (
                            buy_price
                            * shares
                        )


                        # =====================================
                        # BUY手数料
                        # =====================================

                        buy_commission = (
                            self.calculate_commission(
                                buy_price,
                                shares,
                            )
                        )


                        # =====================================
                        # BUYスリッページ
                        # =====================================

                        buy_slippage = (
                            self
                            .calculate_buy_slippage_cost(
                                market_open,
                                buy_price,
                                shares,
                            )
                        )


                        # =====================================
                        # 実際に必要な現金
                        # =====================================

                        required_cash = (
                            entry_value
                            + buy_commission
                        )


                        # =====================================
                        # 最終資金チェック
                        # =====================================

                        if (
                            shares <= 0
                            or required_cash
                            > cash + 1e-9
                        ):

                            can_trade = False


                        # =====================================
                        # BUY成立
                        # =====================================

                        if can_trade:

                            cash_before_buy = cash

                            cash -= required_cash


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
                                    probability,

                                "entry_score":
                                    score,

                                "holding_days":
                                    0,

                                "cash_before_buy":
                                    cash_before_buy,

                                "cash_after_buy":
                                    cash,
                            }


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

                                "Required_Cash":
                                    required_cash,

                                "Cash_After":
                                    cash,
                            })


                        # =====================================
                        # BUY見送り
                        # =====================================

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
                                    probability,

                                "Score":
                                    score,

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

                                "Execution_Price":
                                    buy_price,

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


                        # =====================================
                        # SELL約定価格
                        # =====================================

                        sell_price = (
                            self.calculate_sell_price(
                                market_open
                            )
                        )


                        # =====================================
                        # SELL代金
                        # =====================================

                        sell_value = (
                            sell_price
                            * shares
                        )


                        # =====================================
                        # SELL手数料
                        # =====================================

                        sell_commission = (
                            self.calculate_commission(
                                sell_price,
                                shares,
                            )
                        )


                        # =====================================
                        # 売却手取額
                        # =====================================

                        net_sell_proceeds = (
                            sell_value
                            - sell_commission
                        )


                        # =====================================
                        # 現金へ戻す
                        # =====================================

                        cash += (
                            net_sell_proceeds
                        )


                        # =====================================
                        # 取引レコード
                        # =====================================

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

                            "Trade_Value":
                                sell_value,

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
            # B. 当日終値時点のAI情報
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
            # C. 保有中 → SELL判定
            # =================================================

            if (
                position is not None
                and pending_order is None
            ):

                if (
                    current_date
                    > position[
                        "entry_date"
                    ]
                ):

                    position[
                        "holding_days"
                    ] += 1


                if feature_row is not None:

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
                                position[
                                    "holding_days"
                                ],
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

                                "Probability_Up":
                                    probability,

                                "Reason":
                                    pending_order[
                                        "reason"
                                    ],
                            })


            # =================================================
            # D. ノーポジション → BUY判定
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
            # E. 日次資産評価
            # =================================================

            if position is not None:

                # ---------------------------------------------
                # 未決済ポジションは終値で時価評価
                #
                # BUY手数料は既にcashから引かれているので
                # 二重控除しない。
                # ---------------------------------------------

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
        # 最終日強制決済
        # ====================================================

        if position is not None:

            final_date = trading_dates[-1]

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


                # =============================================
                # 最終Closeを市場価格として
                # SELLスリッページ適用
                # =============================================

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


                net_sell_proceeds = (
                    sell_value
                    - sell_commission
                )


                cash += (
                    net_sell_proceeds
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
                    )
                )


                trades.append(
                    trade_record
                )


                order_records.append({

                    "Date":
                        final_date,

                    "Signal_Date":
                        final_date,

                    "Event":
                        "FINAL_SELL",

                    "Market_Price":
                        final_close,

                    "Execution_Price":
                        sell_price,

                    "Shares":
                        shares,

                    "Trade_Value":
                        sell_value,

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


                # =============================================
                # 最終Equityを修正
                # =============================================

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
        # DataFrame
        # ====================================================

        self.trades = pd.DataFrame(
            trades
        )


        self.equity_curve = pd.DataFrame(
            equity_records
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


        # ====================================================
        # 会計指標
        # ====================================================

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
    # 最大ドローダウン
    # ========================================================

    def _calculate_max_drawdown(
        self,
    ):

        if self.equity_curve.empty:
            return 0.0


        if (
            "Total_Equity"
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

        running_max = running_max[
            valid
        ]


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
    # 会計・成績計算
    # ========================================================

    def calculate_metrics(
        self,
        final_cash,
    ):

        final_capital = float(
            final_cash
        )


        # ====================================================
        # 経路A
        # 最終資産 - 初期資金
        # ====================================================

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

            accounting_difference = (
                asset_profit
            )


            accounting_ok = (
                abs(
                    accounting_difference
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

                # ============================================
                # コスト
                # ============================================

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

                # ============================================
                # 会計
                # ============================================

                "trade_profit_sum":
                    0.0,

                "win_loss_profit":
                    0.0,

                "asset_profit":
                    asset_profit,

                "accounting_difference":
                    accounting_difference,

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
        # Net Profit
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
            len(
                profits
            )
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


        # ====================================================
        # 経路B
        # 全Net_Profit合計
        # ====================================================

        trade_profit_sum = float(
            profits.sum()
        )


        # ====================================================
        # 経路C
        # 勝ち合計 - 負け合計
        # ====================================================

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
        # Profit Factor
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
        # 手数料集計
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
        # スリッページ集計
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


        # ====================================================
        # 総取引コスト
        # ====================================================

        total_trading_cost = (
            total_commission
            + total_slippage
        )


        # ====================================================
        # 保有日数
        # ====================================================

        holding_days = pd.to_numeric(
            self.trades[
                "Holding_Days"
            ],
            errors="coerce",
        )


        if holding_days.notna().any():

            average_holding_days = float(
                holding_days.mean()
            )

        else:

            average_holding_days = None


        # ====================================================
        # 会計監査1
        #
        # 資産増減
        # -
        # 全取引純損益
        # ====================================================

        accounting_difference = (
            asset_profit
            - trade_profit_sum
        )


        # ====================================================
        # 会計監査2
        #
        # NetProfit合計
        # -
        # 勝ち利益＋負け利益
        # ====================================================

        win_loss_difference = (
            trade_profit_sum
            - win_loss_profit
        )


        # ====================================================
        # 会計監査3
        #
        # 各取引
        # NetProfit
        # vs
        # 市場損益 - コスト
        # ====================================================

        if (
            "Trade_Audit_Difference"
            in self.trades.columns
        ):

            trade_audit_diff = (
                pd.to_numeric(
                    self.trades[
                        "Trade_Audit_Difference"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .abs()
            )


            trade_audit_max_difference = float(
                trade_audit_diff.max()
            )


            all_trade_audit_ok = bool(
                (
                    trade_audit_diff
                    < 0.01
                ).all()
            )

        else:

            trade_audit_max_difference = (
                float("inf")
            )

            all_trade_audit_ok = False


        # ====================================================
        # 最終会計判定
        # ====================================================

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


        # ====================================================
        # metrics
        # ====================================================

        return {

            # ================================================
            # 資産
            # ================================================

            "initial_capital":
                self.initial_capital,

            "final_capital":
                final_capital,

            "total_profit":
                asset_profit,

            "total_return":
                total_return,

            # ================================================
            # 取引
            # ================================================

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

            # ================================================
            # 全取引コスト
            # ================================================

            "total_trading_cost":
                total_trading_cost,

            # ================================================
            # 会計監査
            # ================================================

            # 全NetProfit合計
            "trade_profit_sum":
                trade_profit_sum,

            # 勝ち - 負け
            "win_loss_profit":
                win_loss_profit,

            # 最終資産 - 初期資金
            "asset_profit":
                asset_profit,

            # 資産増減 vs NetProfit
            "accounting_difference":
                accounting_difference,

            # NetProfit vs 勝敗集計
            "win_loss_difference":
                win_loss_difference,

            # 各取引最大誤差
            "trade_audit_max_difference":
                trade_audit_max_difference,

            # 全取引個別監査
            "all_trade_audit_ok":
                all_trade_audit_ok,

            # 最終判定
            "accounting_ok":
                accounting_ok,

            # ================================================
            # その他
            # ================================================

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

        return self.equity_curve.copy()


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
