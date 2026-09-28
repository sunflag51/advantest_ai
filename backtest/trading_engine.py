# ============================================================
# backtest/trading_engine.py
#
# 本格戦略バックテストエンジン
#
# Version 3
#
# 改善点
# ------------------------------------------------------------
# ・BUY / SELL はシグナル翌営業日の始値で約定
# ・未来情報を使わない
# ・買いスリッページを明示計算
# ・売りスリッページを明示計算
# ・買い手数料を明示計算
# ・売り手数料を明示計算
# ・純損益を約定金額ベースで厳密計算
# ・RiskManagerのcommissionキーに依存しない
# ・売買履歴に全コストを記録
# ・資産曲線を毎日記録
# ・資金不足BUYを記録
# ・注文ログを記録
# ・最終日に保有株があれば強制決済
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
        # 基本
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
        # 保存
        # ====================================================

        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.metrics = {}

        self.skipped_entries = pd.DataFrame()

        self.order_log = pd.DataFrame()


    # ========================================================
    # float安全変換
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
    def _prepare_index(data):

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

        result = result.sort_index()

        return result


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        stock_data,
        ai_data,
        walk_results,
    ):

        if stock_data is None:
            raise ValueError(
                "stock_data がありません。"
            )

        if ai_data is None:
            raise ValueError(
                "ai_data がありません。"
            )

        if walk_results is None:
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


        # ====================================================
        # 株価必須列
        # ====================================================

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


        # ====================================================
        # 予測必須列
        # ====================================================

        if (
            "Probability_Up"
            not in predictions.columns
        ):

            raise ValueError(
                "walk_results に "
                "Probability_Up がありません。"
            )


        # ====================================================
        # 数値化
        # ====================================================

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


        if stock.empty:

            raise ValueError(
                "有効な株価データがありません。"
            )


        if predictions.empty:

            raise ValueError(
                "有効なAI予測データがありません。"
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

        value = predictions.loc[
            date,
            "Probability_Up",
        ]

        return self._safe_float(
            value
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
    #
    # 買いでは不利方向に価格を上げる
    # ========================================================

    def calculate_buy_price(
        self,
        market_open,
    ):

        market_open = float(
            market_open
        )

        return (
            market_open
            * (
                1.0
                + self.slippage_rate
            )
        )


    # ========================================================
    # SELL約定価格
    #
    # 売りでは不利方向に価格を下げる
    # ========================================================

    def calculate_sell_price(
        self,
        market_open,
    ):

        market_open = float(
            market_open
        )

        return (
            market_open
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

        trade_value = (
            float(price)
            * int(shares)
        )

        return (
            trade_value
            * self.commission_rate
        )


    # ========================================================
    # Exit安全評価
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
                "action": "HOLD",
                "reason": "NO_CLOSE",
            }


        # ====================================================
        # AI確率あり
        # ====================================================

        if probability is not None:

            try:

                return (
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

            except TypeError:

                try:

                    return (
                        self.exit_strategy
                        .evaluate(
                            row,
                            probability,
                            entry_price,
                            holding_days,
                        )
                    )

                except Exception:
                    pass

            except Exception:
                pass


        # ====================================================
        # AI確率がない日でも
        # ハードExitだけは判定
        # ====================================================

        return_rate = (
            current_price
            / entry_price
            - 1.0
        )


        stop_loss_rate = getattr(
            self.exit_strategy,
            "stop_loss_rate",
            0.05,
        )


        take_profit_rate = getattr(
            self.exit_strategy,
            "take_profit_rate",
            0.10,
        )


        max_holding_days = getattr(
            self.exit_strategy,
            "max_holding_days",
            10,
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


        # ====================================================
        # 初期化
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


        # ====================================================
        # Walk Forward開始日
        # ====================================================

        prediction_start = (
            predictions.index.min()
        )


        trading_dates = stock.index[
            stock.index
            >= prediction_start
        ]


        if len(trading_dates) == 0:

            raise ValueError(
                "バックテスト可能な日付がありません。"
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
            # A. 始値でPending注文を約定
            # =================================================

            if pending_order is not None:

                intended_date = (
                    pending_order.get(
                        "execution_date"
                    )
                )


                if (
                    intended_date is None
                    or current_date
                    >= intended_date
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


                        # -------------------------------------
                        # 実際のBUY約定価格
                        # -------------------------------------

                        buy_price = (
                            self.calculate_buy_price(
                                market_open
                            )
                        )


                        # -------------------------------------
                        # RiskManagerは株数決定に使用
                        #
                        # market_priceには始値を渡す。
                        # RiskManager内部にもslippage設定があるため、
                        # RiskManagerが計算する株数は安全側。
                        #
                        # 実際の会計計算は下でTradingEngine自身が行う。
                        # -------------------------------------

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


                        # -------------------------------------
                        # 念のため100株単位へ丸める
                        # -------------------------------------

                        shares = (
                            shares
                            // self.lot_size
                            * self.lot_size
                        )


                        # -------------------------------------
                        # 実際の買い代金
                        # -------------------------------------

                        buy_value = (
                            buy_price
                            * shares
                        )


                        # -------------------------------------
                        # 買い手数料
                        # -------------------------------------

                        buy_commission = (
                            self.calculate_commission(
                                buy_price,
                                shares,
                            )
                        )


                        # -------------------------------------
                        # 必要現金
                        # -------------------------------------

                        required_cash = (
                            buy_value
                            + buy_commission
                        )


                        # -------------------------------------
                        # 最終的な現金チェック
                        # -------------------------------------

                        if (
                            shares <= 0
                            or required_cash
                            > cash
                        ):

                            can_trade = False


                        if can_trade:

                            cash_before = cash


                            # ---------------------------------
                            # 現金から
                            # 購入代金＋買い手数料を引く
                            # ---------------------------------

                            cash -= required_cash


                            # ---------------------------------
                            # BUYスリッページ額
                            #
                            # 実際の始値との差額
                            # ---------------------------------

                            buy_slippage_cost = (
                                (
                                    buy_price
                                    - market_open
                                )
                                * shares
                            )


                            position = {

                                "entry_date":
                                    current_date,

                                "signal_date":
                                    signal_date,

                                "shares":
                                    shares,

                                "entry_market_open":
                                    market_open,

                                "entry_price":
                                    buy_price,

                                "entry_value":
                                    buy_value,

                                "buy_commission":
                                    buy_commission,

                                "buy_slippage_cost":
                                    buy_slippage_cost,

                                "entry_probability":
                                    signal_probability,

                                "entry_score":
                                    signal_score,

                                "holding_days":
                                    0,

                                "cash_before_buy":
                                    cash_before,

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

                                "Market_Open":
                                    market_open,

                                "Execution_Price":
                                    buy_price,

                                "Shares":
                                    shares,

                                "Trade_Value":
                                    buy_value,

                                "Commission":
                                    buy_commission,

                                "Slippage_Cost":
                                    buy_slippage_cost,

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

                                "Market_Open":
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

                                "Market_Open":
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


                        # -------------------------------------
                        # SELL約定価格
                        # -------------------------------------

                        sell_price = (
                            self.calculate_sell_price(
                                market_open
                            )
                        )


                        shares = int(
                            position[
                                "shares"
                            ]
                        )


                        # -------------------------------------
                        # 売却代金
                        # -------------------------------------

                        sell_value = (
                            sell_price
                            * shares
                        )


                        # -------------------------------------
                        # 売り手数料
                        # -------------------------------------

                        sell_commission = (
                            self.calculate_commission(
                                sell_price,
                                shares,
                            )
                        )


                        # -------------------------------------
                        # SELLスリッページ
                        # -------------------------------------

                        sell_slippage_cost = (
                            (
                                market_open
                                - sell_price
                            )
                            * shares
                        )


                        # -------------------------------------
                        # 売却後受取金
                        # -------------------------------------

                        net_sell_proceeds = (
                            sell_value
                            - sell_commission
                        )


                        # -------------------------------------
                        # 現金へ戻す
                        # -------------------------------------

                        cash += (
                            net_sell_proceeds
                        )


                        # -------------------------------------
                        # 投入総コスト
                        #
                        # BUY代金＋BUY手数料
                        # -------------------------------------

                        total_entry_cost = (
                            position[
                                "entry_value"
                            ]
                            + position[
                                "buy_commission"
                            ]
                        )


                        # -------------------------------------
                        # 純損益
                        #
                        # 売却手取額
                        # -
                        # 購入総コスト
                        # -------------------------------------

                        net_profit = (
                            net_sell_proceeds
                            - total_entry_cost
                        )


                        # -------------------------------------
                        # 純損益率
                        # -------------------------------------

                        if total_entry_cost > 0:

                            net_return = (
                                net_profit
                                / total_entry_cost
                            )

                        else:

                            net_return = 0.0


                        # -------------------------------------
                        # 売買手数料合計
                        # -------------------------------------

                        total_commission = (
                            position[
                                "buy_commission"
                            ]
                            + sell_commission
                        )


                        # -------------------------------------
                        # スリッページ合計
                        # -------------------------------------

                        total_slippage = (
                            position[
                                "buy_slippage_cost"
                            ]
                            + sell_slippage_cost
                        )


                        # -------------------------------------
                        # 市場価格だけで見た損益
                        #
                        # スリッページ・手数料なし
                        # -------------------------------------

                        gross_market_profit = (
                            (
                                market_open
                                - position[
                                    "entry_market_open"
                                ]
                            )
                            * shares
                        )


                        trades.append({

                            "Entry_Signal_Date":
                                position[
                                    "signal_date"
                                ],

                            "Entry_Date":
                                position[
                                    "entry_date"
                                ],

                            "Exit_Signal_Date":
                                signal_date,

                            "Exit_Date":
                                current_date,

                            "Shares":
                                shares,

                            "Entry_Market_Open":
                                position[
                                    "entry_market_open"
                                ],

                            "Entry_Price":
                                position[
                                    "entry_price"
                                ],

                            "Exit_Market_Open":
                                market_open,

                            "Exit_Price":
                                sell_price,

                            "Entry_Value":
                                position[
                                    "entry_value"
                                ],

                            "Exit_Value":
                                sell_value,

                            "Buy_Commission":
                                position[
                                    "buy_commission"
                                ],

                            "Sell_Commission":
                                sell_commission,

                            "Total_Commission":
                                total_commission,

                            "Buy_Slippage":
                                position[
                                    "buy_slippage_cost"
                                ],

                            "Sell_Slippage":
                                sell_slippage_cost,

                            "Total_Slippage":
                                total_slippage,

                            "Gross_Market_Profit":
                                gross_market_profit,

                            "Net_Profit":
                                net_profit,

                            "Net_Return":
                                net_return,

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
                                False,

                            "Cash_After_Exit":
                                cash,
                        })


                        order_records.append({

                            "Date":
                                current_date,

                            "Signal_Date":
                                signal_date,

                            "Event":
                                "SELL_FILLED",

                            "Market_Open":
                                market_open,

                            "Execution_Price":
                                sell_price,

                            "Shares":
                                shares,

                            "Trade_Value":
                                sell_value,

                            "Commission":
                                sell_commission,

                            "Slippage_Cost":
                                sell_slippage_cost,

                            "Net_Profit":
                                net_profit,

                            "Cash_After":
                                cash,

                            "Reason":
                                exit_reason,
                        })


                        position = None

                        pending_order = None


            # =================================================
            # B. 当日終値時点の情報
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

                # ---------------------------------------------
                # Entry日より後の日だけ保有日数加算
                # ---------------------------------------------

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

                market_position_value = (
                    market_close
                    * position[
                        "shares"
                    ]
                )

            else:

                market_position_value = 0.0


            total_equity = (
                cash
                + market_position_value
            )


            equity_records.append({

                "Date":
                    current_date,

                "Cash":
                    cash,

                "Position_Value":
                    market_position_value,

                "Total_Equity":
                    total_equity,

                "Shares":
                    (
                        position[
                            "shares"
                        ]
                        if position
                        is not None
                        else 0
                    ),

                "Close":
                    market_close,
            })


        # ====================================================
        # 最終日にポジションが残っている場合
        #
        # 最終日は次営業日がないのでCloseで強制決済。
        # SELL側スリッページを適用。
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


                # ---------------------------------------------
                # 最終Closeから売りスリッページ
                # ---------------------------------------------

                sell_price = (
                    final_close
                    * (
                        1.0
                        - self.slippage_rate
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


                sell_slippage_cost = (
                    (
                        final_close
                        - sell_price
                    )
                    * shares
                )


                net_sell_proceeds = (
                    sell_value
                    - sell_commission
                )


                cash += (
                    net_sell_proceeds
                )


                total_entry_cost = (
                    position[
                        "entry_value"
                    ]
                    + position[
                        "buy_commission"
                    ]
                )


                net_profit = (
                    net_sell_proceeds
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
                    position[
                        "buy_commission"
                    ]
                    + sell_commission
                )


                total_slippage = (
                    position[
                        "buy_slippage_cost"
                    ]
                    + sell_slippage_cost
                )


                gross_market_profit = (
                    (
                        final_close
                        - position[
                            "entry_market_open"
                        ]
                    )
                    * shares
                )


                trades.append({

                    "Entry_Signal_Date":
                        position[
                            "signal_date"
                        ],

                    "Entry_Date":
                        position[
                            "entry_date"
                        ],

                    "Exit_Signal_Date":
                        final_date,

                    "Exit_Date":
                        final_date,

                    "Shares":
                        shares,

                    "Entry_Market_Open":
                        position[
                            "entry_market_open"
                        ],

                    "Entry_Price":
                        position[
                            "entry_price"
                        ],

                    "Exit_Market_Open":
                        final_close,

                    "Exit_Price":
                        sell_price,

                    "Entry_Value":
                        position[
                            "entry_value"
                        ],

                    "Exit_Value":
                        sell_value,

                    "Buy_Commission":
                        position[
                            "buy_commission"
                        ],

                    "Sell_Commission":
                        sell_commission,

                    "Total_Commission":
                        total_commission,

                    "Buy_Slippage":
                        position[
                            "buy_slippage_cost"
                        ],

                    "Sell_Slippage":
                        sell_slippage_cost,

                    "Total_Slippage":
                        total_slippage,

                    "Gross_Market_Profit":
                        gross_market_profit,

                    "Net_Profit":
                        net_profit,

                    "Net_Return":
                        net_return,

                    "Holding_Days":
                        position[
                            "holding_days"
                        ],

                    "Entry_Probability":
                        position[
                            "entry_probability"
                        ],

                    "Exit_Probability":
                        None,

                    "Entry_Score":
                        position[
                            "entry_score"
                        ],

                    "Exit_Reason":
                        "FINAL_DATA_EXIT",

                    "Forced_Exit":
                        True,

                    "Cash_After_Exit":
                        cash,
                })


                order_records.append({

                    "Date":
                        final_date,

                    "Signal_Date":
                        final_date,

                    "Event":
                        "FINAL_SELL",

                    "Market_Open":
                        final_close,

                    "Execution_Price":
                        sell_price,

                    "Shares":
                        shares,

                    "Trade_Value":
                        sell_value,

                    "Commission":
                        sell_commission,

                    "Slippage_Cost":
                        sell_slippage_cost,

                    "Net_Profit":
                        net_profit,

                    "Cash_After":
                        cash,

                    "Reason":
                        "FINAL_DATA_EXIT",
                })


                position = None


                # ---------------------------------------------
                # 最終日のEquityを現金値へ修正
                # ---------------------------------------------

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


        # ====================================================
        # 指標
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
    # 指標計算
    # ========================================================

    def calculate_metrics(
        self,
        final_cash,
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
            if self.initial_capital > 0
            else 0.0
        )


        # ====================================================
        # 取引なし
        # ====================================================

        if self.trades.empty:

            return {

                "initial_capital":
                    self.initial_capital,

                "final_capital":
                    final_capital,

                "total_profit":
                    total_profit,

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

                "total_commission":
                    0.0,

                "total_slippage":
                    0.0,

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


        trade_count = len(
            profits
        )


        win_rate = (
            wins
            / trade_count
            if trade_count > 0
            else None
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

        total_commission = float(
            pd.to_numeric(
                self.trades[
                    "Total_Commission"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
        )


        # ====================================================
        # スリッページ
        # ====================================================

        total_slippage = float(
            pd.to_numeric(
                self.trades[
                    "Total_Slippage"
                ],
                errors="coerce",
            ).fillna(0.0).sum()
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


        average_holding_days = (
            float(
                holding_days.mean()
            )
            if holding_days.notna().any()
            else None
        )


        # ====================================================
        # 重要整合性チェック
        #
        # 各トレード純損益の合計と
        # 最終資産差額は一致するはず。
        # ====================================================

        trade_profit_sum = float(
            profits.sum()
        )


        accounting_difference = (
            total_profit
            - trade_profit_sum
        )


        return {

            "initial_capital":
                self.initial_capital,

            "final_capital":
                final_capital,

            "total_profit":
                total_profit,

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

            "total_commission":
                total_commission,

            "total_slippage":
                total_slippage,

            "trade_profit_sum":
                trade_profit_sum,

            "accounting_difference":
                accounting_difference,

            "accounting_ok":
                abs(
                    accounting_difference
                ) < 0.01,

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

        return self.skipped_entries.copy()


    def get_order_log(
        self,
    ):

        return self.order_log.copy()


# ============================================================
# 簡易実行関数
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
