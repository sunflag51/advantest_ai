# ============================================================
# アドバンテスト AI売買システム
# backtest/trading_engine.py
#
# 本格戦略バックテストエンジン 改良完全版
#
# 重要な売買ルール
# ------------------------------------------------------------
# 1. 当日の終値が確定した後にシグナル判定
# 2. BUYシグナル → 翌営業日の始値で購入
# 3. SELLシグナル → 翌営業日の始値で売却
# 4. 同じ日の終値を見て、その終値で売買しない
# 5. 売買手数料・スリッページを考慮
# 6. EntryStrategy / ExitStrategy / RiskManager を統合
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

        # Entry
        entry_minimum_score=6,
        entry_minimum_probability=0.55,
        entry_strong_probability=0.60,

        # Exit
        stop_loss_rate=0.05,
        take_profit_rate=0.10,
        ai_exit_probability=0.45,
        max_holding_days=10,
        minimum_exit_score=3,

        # Risk
        risk_per_trade=0.01,
        max_position_rate=0.50,

        # Cost
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
        # EntryStrategy
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
        # ExitStrategy
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
        # RiskManager
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
    # インデックス整理
    # ========================================================

    @staticmethod
    def _prepare_index(data):

        if data is None:

            return pd.DataFrame()


        if not isinstance(
            data,
            pd.DataFrame,
        ):

            return pd.DataFrame()


        if data.empty:

            return pd.DataFrame()


        df = data.copy()


        try:

            df.index = pd.to_datetime(
                df.index
            )

        except Exception:

            pass


        # timezone除去
        try:

            if df.index.tz is not None:

                df.index = (
                    df.index.tz_localize(
                        None
                    )
                )

        except Exception:

            pass


        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]


        df = df.sort_index()


        return df


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
                pd.Series,
            ):

                values = (
                    pd.to_numeric(
                        value,
                        errors="coerce",
                    )
                    .dropna()
                )


                if values.empty:

                    return default


                return float(
                    values.iloc[-1]
                )


            number = float(
                value
            )


            if not np.isfinite(
                number
            ):

                return default


            return number


        except Exception:

            return default


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        stock_data,
        ai_data,
        walk_results,
    ):

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
        # 株価データ確認
        # ====================================================

        required_stock_columns = [
            "Open",
            "High",
            "Low",
            "Close",
        ]


        missing_stock_columns = [
            column

            for column
            in required_stock_columns

            if column
            not in stock.columns
        ]


        if missing_stock_columns:

            raise ValueError(
                "株価データに必要な列がありません: "
                + ", ".join(
                    missing_stock_columns
                )
            )


        # ====================================================
        # Walk Forward確認
        # ====================================================

        if "Probability_Up" not in predictions.columns:

            raise ValueError(
                "walk_results に "
                "Probability_Up 列がありません。"
            )


        # ====================================================
        # 数値化
        # ====================================================

        for column in [
            "Open",
            "High",
            "Low",
            "Close",
        ]:

            stock[column] = (
                pd.to_numeric(
                    stock[column],
                    errors="coerce",
                )
            )


        predictions[
            "Probability_Up"
        ] = (
            pd.to_numeric(
                predictions[
                    "Probability_Up"
                ],
                errors="coerce",
            )
        )


        stock = stock.dropna(
            subset=[
                "Open",
                "Close",
            ]
        )


        if stock.empty:

            raise ValueError(
                "有効な株価データがありません。"
            )


        if predictions.empty:

            raise ValueError(
                "ウォークフォワード予測結果がありません。"
            )


        return (
            stock,
            features,
            predictions,
        )


    # ========================================================
    # AI確率取得
    # ========================================================

    def get_probability(
        self,
        predictions,
        date,
    ):

        if date not in predictions.index:

            return None


        try:

            value = predictions.loc[
                date,
                "Probability_Up"
            ]


            probability = (
                self._safe_float(
                    value
                )
            )


            if probability is None:

                return None


            if (
                probability < 0
                or probability > 1
            ):

                return None


            return probability


        except Exception:

            return None


    # ========================================================
    # 特徴量行取得
    # ========================================================

    def get_feature_row(
        self,
        features,
        date,
    ):

        if date not in features.index:

            return None


        try:

            row = features.loc[
                date
            ]


            # 重複日付等でDataFrameになった場合
            if isinstance(
                row,
                pd.DataFrame,
            ):

                if row.empty:

                    return None


                row = row.iloc[-1]


            return row


        except Exception:

            return None


    # ========================================================
    # 次営業日取得
    # ========================================================

    @staticmethod
    def get_next_trading_date(
        stock_index,
        current_date,
    ):

        try:

            location = (
                stock_index.get_loc(
                    current_date
                )
            )


            # 重複等への保険
            if isinstance(
                location,
                slice,
            ):

                location = (
                    location.stop
                    - 1
                )


            if isinstance(
                location,
                np.ndarray,
            ):

                positions = np.where(
                    location
                )[0]

                if len(
                    positions
                ) == 0:

                    return None

                location = int(
                    positions[-1]
                )


            next_location = (
                int(location)
                + 1
            )


            if (
                next_location
                >= len(stock_index)
            ):

                return None


            return stock_index[
                next_location
            ]


        except Exception:

            return None


    # ========================================================
    # BUY約定価格
    # ========================================================

    def calculate_buy_price(
        self,
        open_price,
    ):

        return (
            float(open_price)
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
        open_price,
    ):

        return (
            float(open_price)
            * (
                1.0
                - self.slippage_rate
            )
        )


    # ========================================================
    # 売却手数料
    # ========================================================

    def calculate_sell_commission(
        self,
        sell_price,
        shares,
    ):

        gross_value = (
            float(sell_price)
            * int(shares)
        )


        return (
            gross_value
            * self.commission_rate
        )


    # ========================================================
    # Exit判定
    # ========================================================

    def evaluate_exit_safely(
        self,
        row,
        probability_up,
        entry_price,
        current_price,
        holding_days,
    ):
        """
        probability_up が存在しない場合でも
        バックテストを停止させない。

        AI確率がある日は通常のExitStrategy。

        AI確率がない日は、
        損切り・利益確定・最大保有期間を
        優先して確認する。
        """

        # ----------------------------------------------------
        # AI確率がある
        # ----------------------------------------------------

        if probability_up is not None:

            return self.exit_strategy.evaluate(
                row=row,
                probability_up=
                    probability_up,

                entry_price=
                    entry_price,

                current_price=
                    current_price,

                holding_days=
                    holding_days,
            )


        # ----------------------------------------------------
        # AI確率がない
        # Hard Exitのみ確認
        # ----------------------------------------------------

        hard_exit = (
            self.exit_strategy
            .evaluate_hard_exit(
                entry_price=
                    entry_price,

                current_price=
                    current_price,

                holding_days=
                    holding_days,
            )
        )


        if hard_exit.get(
            "triggered",
            False,
        ):

            return {
                "action":
                    "SELL",

                "exit_type":
                    hard_exit.get(
                        "exit_type",
                        "HARD_EXIT",
                    ),

                "reason_code":
                    hard_exit.get(
                        "reason_code",
                        "HARD_EXIT",
                    ),

                "summary":
                    hard_exit.get(
                        "summary",
                        "ハードExit条件成立",
                    ),

                "exit_score":
                    0,

                "max_score":
                    7,

                "probability_up":
                    None,

                "holding_days":
                    holding_days,
            }


        return {
            "action":
                "HOLD",

            "exit_type":
                "NONE",

            "reason_code":
                "NO_AI_PROBABILITY",

            "summary":
                "AI確率なし・ハードExit条件なし",

            "exit_score":
                0,

            "max_score":
                7,

            "probability_up":
                None,

            "holding_days":
                holding_days,
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
            stock_data=
                stock_data,

            ai_data=
                ai_data,

            walk_results=
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

        order_log = []


        prediction_start = (
            predictions.index.min()
        )


        trading_dates = stock.index[
            stock.index
            >= prediction_start
        ]


        if len(
            trading_dates
        ) == 0:

            raise ValueError(
                "バックテスト可能な営業日がありません。"
            )


        # ====================================================
        # 日次ループ
        # ====================================================

        for current_date in trading_dates:

            current_row = stock.loc[
                current_date
            ]


            if isinstance(
                current_row,
                pd.DataFrame,
            ):

                current_row = (
                    current_row.iloc[-1]
                )


            open_price = (
                self._safe_float(
                    current_row.get(
                        "Open"
                    )
                )
            )


            close_price = (
                self._safe_float(
                    current_row.get(
                        "Close"
                    )
                )
            )


            if (
                open_price is None
                or close_price is None
            ):

                continue


            # =================================================
            # A. 始値で予約注文を約定
            # =================================================

            if pending_order is not None:

                order_type = (
                    pending_order.get(
                        "type"
                    )
                )


                # =============================================
                # BUY注文
                # =============================================

                if (
                    order_type == "BUY"
                    and position is None
                ):

                    risk_result = (
                        self.risk_manager
                        .evaluate_trade(
                            capital=
                                cash,

                            market_price=
                                open_price,
                        )
                    )


                    can_trade = bool(
                        risk_result.get(
                            "can_trade",
                            False,
                        )
                    )


                    shares = int(
                        risk_result.get(
                            "shares",
                            0,
                        )
                    )


                    if (
                        can_trade
                        and shares
                        >= self.lot_size
                    ):

                        entry_price = (
                            self._safe_float(
                                risk_result.get(
                                    "entry_price"
                                )
                            )
                        )


                        required_cash = (
                            self._safe_float(
                                risk_result.get(
                                    "required_cash"
                                )
                            )
                        )


                        purchase_value = (
                            self._safe_float(
                                risk_result.get(
                                    "purchase_value"
                                )
                            )
                        )


                        buy_commission = (
                            self._safe_float(
                                risk_result.get(
                                    "buy_commission"
                                ),
                                0.0,
                            )
                        )


                        stop_price = (
                            self._safe_float(
                                risk_result.get(
                                    "stop_price"
                                )
                            )
                        )


                        take_profit_price = (
                            self._safe_float(
                                risk_result.get(
                                    "take_profit_price"
                                )
                            )
                        )


                        if (
                            entry_price is not None
                            and required_cash is not None
                            and required_cash <= cash
                        ):

                            cash -= (
                                required_cash
                            )


                            position = {
                                "entry_date":
                                    current_date,

                                "signal_date":
                                    pending_order.get(
                                        "signal_date"
                                    ),

                                "entry_price":
                                    entry_price,

                                "market_open":
                                    open_price,

                                "shares":
                                    shares,

                                "purchase_value":
                                    purchase_value,

                                "buy_commission":
                                    buy_commission,

                                "stop_price":
                                    stop_price,

                                "take_profit_price":
                                    take_profit_price,

                                "entry_probability":
                                    pending_order.get(
                                        "probability_up"
                                    ),

                                "entry_score":
                                    pending_order.get(
                                        "entry_score"
                                    ),

                                "holding_days":
                                    0,
                            }


                            order_log.append(
                                {
                                    "Date":
                                        current_date,

                                    "Order":
                                        "BUY",

                                    "Status":
                                        "FILLED",

                                    "Signal_Date":
                                        pending_order.get(
                                            "signal_date"
                                        ),

                                    "Market_Open":
                                        open_price,

                                    "Execution_Price":
                                        entry_price,

                                    "Shares":
                                        shares,
                                }
                            )


                        else:

                            skipped_entries.append(
                                {
                                    "Signal_Date":
                                        pending_order.get(
                                            "signal_date"
                                        ),

                                    "Execution_Date":
                                        current_date,

                                    "Probability_Up":
                                        pending_order.get(
                                            "probability_up"
                                        ),

                                    "Entry_Score":
                                        pending_order.get(
                                            "entry_score"
                                        ),

                                    "Market_Open":
                                        open_price,

                                    "Reason":
                                        "必要資金を確保できません",
                                }
                            )


                    else:

                        skipped_entries.append(
                            {
                                "Signal_Date":
                                    pending_order.get(
                                        "signal_date"
                                    ),

                                "Execution_Date":
                                    current_date,

                                "Probability_Up":
                                    pending_order.get(
                                        "probability_up"
                                    ),

                                "Entry_Score":
                                    pending_order.get(
                                        "entry_score"
                                    ),

                                "Market_Open":
                                    open_price,

                                "Reason":
                                    risk_result.get(
                                        "reason",
                                        "資金管理条件で見送り",
                                    ),
                            }
                        )


                        order_log.append(
                            {
                                "Date":
                                    current_date,

                                "Order":
                                    "BUY",

                                "Status":
                                    "SKIPPED",

                                "Signal_Date":
                                    pending_order.get(
                                        "signal_date"
                                    ),

                                "Market_Open":
                                    open_price,

                                "Execution_Price":
                                    None,

                                "Shares":
                                    0,
                            }
                        )


                    pending_order = None


                # =============================================
                # SELL注文
                # =============================================

                elif (
                    order_type == "SELL"
                    and position is not None
                ):

                    sell_price = (
                        self.calculate_sell_price(
                            open_price
                        )
                    )


                    shares = int(
                        position[
                            "shares"
                        ]
                    )


                    gross_sell_value = (
                        sell_price
                        * shares
                    )


                    sell_commission = (
                        self.calculate_sell_commission(
                            sell_price=
                                sell_price,

                            shares=
                                shares,
                        )
                    )


                    net_sell_value = (
                        gross_sell_value
                        - sell_commission
                    )


                    cash += (
                        net_sell_value
                    )


                    entry_cost = (
                        position[
                            "purchase_value"
                        ]
                        + position[
                            "buy_commission"
                        ]
                    )


                    profit = (
                        net_sell_value
                        - entry_cost
                    )


                    if (
                        entry_cost > 0
                    ):

                        profit_rate = (
                            profit
                            / entry_cost
                        )

                    else:

                        profit_rate = 0.0


                    trades.append(
                        {
                            "Entry_Signal_Date":
                                position.get(
                                    "signal_date"
                                ),

                            "Entry_Date":
                                position[
                                    "entry_date"
                                ],

                            "Exit_Signal_Date":
                                pending_order.get(
                                    "signal_date"
                                ),

                            "Exit_Date":
                                current_date,

                            "Entry_Price":
                                position[
                                    "entry_price"
                                ],

                            "Exit_Price":
                                sell_price,

                            "Shares":
                                shares,

                            "Entry_Probability":
                                position.get(
                                    "entry_probability"
                                ),

                            "Entry_Score":
                                position.get(
                                    "entry_score"
                                ),

                            "Exit_Probability":
                                pending_order.get(
                                    "probability_up"
                                ),

                            "Exit_Score":
                                pending_order.get(
                                    "exit_score"
                                ),

                            "Exit_Reason":
                                pending_order.get(
                                    "reason_code",
                                    "SELL",
                                ),

                            "Holding_Days":
                                position.get(
                                    "holding_days",
                                    0,
                                ),

                            "Purchase_Value":
                                position[
                                    "purchase_value"
                                ],

                            "Gross_Sell_Value":
                                gross_sell_value,

                            "Buy_Commission":
                                position[
                                    "buy_commission"
                                ],

                            "Sell_Commission":
                                sell_commission,

                            "Total_Commission":
                                (
                                    position[
                                        "buy_commission"
                                    ]
                                    + sell_commission
                                ),

                            "Profit":
                                profit,

                            "Profit_Rate":
                                profit_rate,

                            "Forced_Exit":
                                False,
                        }
                    )


                    order_log.append(
                        {
                            "Date":
                                current_date,

                            "Order":
                                "SELL",

                            "Status":
                                "FILLED",

                            "Signal_Date":
                                pending_order.get(
                                    "signal_date"
                                ),

                            "Market_Open":
                                open_price,

                            "Execution_Price":
                                sell_price,

                            "Shares":
                                shares,
                        }
                    )


                    position = None

                    pending_order = None


                else:

                    # 状態と注文が一致しない場合は破棄
                    pending_order = None


            # =================================================
            # B. 当日終値時点のAI確率・特徴量
            # =================================================

            probability_up = (
                self.get_probability(
                    predictions=
                        predictions,

                    date=
                        current_date,
                )
            )


            feature_row = (
                self.get_feature_row(
                    features=
                        features,

                    date=
                        current_date,
                )
            )


            # =================================================
            # C. 終値確定後にSELL判定
            # =================================================

            if (
                position is not None
                and pending_order is None
            ):

                # エントリー日を0日として、
                # 翌営業日から保有日数を増やす
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

                            probability_up=
                                probability_up,

                            entry_price=
                                position[
                                    "entry_price"
                                ],

                            current_price=
                                close_price,

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

                                "probability_up":
                                    probability_up,

                                "exit_score":
                                    exit_result.get(
                                        "exit_score",
                                        0,
                                    ),

                                "reason_code":
                                    exit_result.get(
                                        "reason_code",
                                        "SELL",
                                    ),

                                "summary":
                                    exit_result.get(
                                        "summary",
                                        "",
                                    ),
                            }


                            order_log.append(
                                {
                                    "Date":
                                        current_date,

                                    "Order":
                                        "SELL",

                                    "Status":
                                        "SIGNAL",

                                    "Signal_Date":
                                        current_date,

                                    "Market_Open":
                                        None,

                                    "Execution_Price":
                                        None,

                                    "Shares":
                                        position[
                                            "shares"
                                        ],
                                }
                            )


            # =================================================
            # D. 終値確定後にBUY判定
            # =================================================

            elif (
                position is None
                and pending_order is None
                and probability_up is not None
                and feature_row is not None
            ):

                entry_result = (
                    self.entry_strategy.evaluate(
                        row=
                            feature_row,

                        probability_up=
                            probability_up,
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

                            "probability_up":
                                probability_up,

                            "entry_score":
                                entry_result.get(
                                    "score",
                                    0,
                                ),

                            "summary":
                                entry_result.get(
                                    "summary",
                                    "",
                                ),
                        }


                        order_log.append(
                            {
                                "Date":
                                    current_date,

                                "Order":
                                    "BUY",

                                "Status":
                                    "SIGNAL",

                                "Signal_Date":
                                    current_date,

                                "Market_Open":
                                    None,

                                "Execution_Price":
                                    None,

                                "Shares":
                                    0,
                            }
                        )


            # =================================================
            # E. 当日資産評価
            # =================================================

            market_value = 0.0


            if position is not None:

                market_value = (
                    close_price
                    * position[
                        "shares"
                    ]
                )


            total_equity = (
                cash
                + market_value
            )


            equity_records.append(
                {
                    "Date":
                        current_date,

                    "Cash":
                        cash,

                    "Market_Value":
                        market_value,

                    "Total_Equity":
                        total_equity,

                    "Position":
                        (
                            position[
                                "shares"
                            ]

                            if position
                            is not None

                            else 0
                        ),

                    "Pending_Order":
                        (
                            pending_order.get(
                                "type"
                            )

                            if pending_order
                            is not None

                            else None
                        ),
                }
            )


        # ====================================================
        # 最終日処理
        #
        # 最終日にまだポジションが残っている場合は
        # 最終終値で強制決済する。
        #
        # これは翌営業日の始値がデータ内に存在しないため。
        # Forced_Exit=Trueとして通常売却と区別する。
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

                sell_price = (
                    final_close
                    * (
                        1.0
                        - self.slippage_rate
                    )
                )


                shares = int(
                    position[
                        "shares"
                    ]
                )


                gross_sell_value = (
                    sell_price
                    * shares
                )


                sell_commission = (
                    self.calculate_sell_commission(
                        sell_price=
                            sell_price,

                        shares=
                            shares,
                    )
                )


                net_sell_value = (
                    gross_sell_value
                    - sell_commission
                )


                cash += (
                    net_sell_value
                )


                entry_cost = (
                    position[
                        "purchase_value"
                    ]
                    + position[
                        "buy_commission"
                    ]
                )


                profit = (
                    net_sell_value
                    - entry_cost
                )


                if entry_cost > 0:

                    profit_rate = (
                        profit
                        / entry_cost
                    )

                else:

                    profit_rate = 0.0


                trades.append(
                    {
                        "Entry_Signal_Date":
                            position.get(
                                "signal_date"
                            ),

                        "Entry_Date":
                            position[
                                "entry_date"
                            ],

                        "Exit_Signal_Date":
                            final_date,

                        "Exit_Date":
                            final_date,

                        "Entry_Price":
                            position[
                                "entry_price"
                            ],

                        "Exit_Price":
                            sell_price,

                        "Shares":
                            shares,

                        "Entry_Probability":
                            position.get(
                                "entry_probability"
                            ),

                        "Entry_Score":
                            position.get(
                                "entry_score"
                            ),

                        "Exit_Probability":
                            None,

                        "Exit_Score":
                            None,

                        "Exit_Reason":
                            "FINAL_DATA_EXIT",

                        "Holding_Days":
                            position.get(
                                "holding_days",
                                0,
                            ),

                        "Purchase_Value":
                            position[
                                "purchase_value"
                            ],

                        "Gross_Sell_Value":
                            gross_sell_value,

                        "Buy_Commission":
                            position[
                                "buy_commission"
                            ],

                        "Sell_Commission":
                            sell_commission,

                        "Total_Commission":
                            (
                                position[
                                    "buy_commission"
                                ]
                                + sell_commission
                            ),

                        "Profit":
                            profit,

                        "Profit_Rate":
                            profit_rate,

                        "Forced_Exit":
                            True,
                    }
                )


                order_log.append(
                    {
                        "Date":
                            final_date,

                        "Order":
                            "SELL",

                        "Status":
                            "FORCED_FINAL_CLOSE",

                        "Signal_Date":
                            final_date,

                        "Market_Open":
                            None,

                        "Execution_Price":
                            sell_price,

                        "Shares":
                            shares,
                    }
                )


                position = None


                # --------------------------------------------
                # 最終日の資産記録を更新
                # --------------------------------------------

                if equity_records:

                    equity_records[-1][
                        "Cash"
                    ] = cash

                    equity_records[-1][
                        "Market_Value"
                    ] = 0.0

                    equity_records[-1][
                        "Total_Equity"
                    ] = cash

                    equity_records[-1][
                        "Position"
                    ] = 0

                    equity_records[-1][
                        "Pending_Order"
                    ] = None


        # ====================================================
        # DataFrame化
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
                order_log
            )
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
                .set_index(
                    "Date"
                )
                .sort_index()
            )


        # ====================================================
        # 指標計算
        # ====================================================

        self.metrics = (
            self.calculate_metrics(
                final_cash=
                    cash
            )
        )


        return (
            self.trades,
            self.equity_curve,
            self.metrics,
        )


    # ========================================================
    # 成績計算
    # ========================================================

    def calculate_metrics(
        self,
        final_cash=None,
    ):

        # ====================================================
        # 最終資産
        # ====================================================

        if final_cash is None:

            if (
                self.equity_curve is not None
                and not self.equity_curve.empty
                and "Total_Equity"
                in self.equity_curve.columns
            ):

                final_capital = (
                    self._safe_float(
                        self.equity_curve[
                            "Total_Equity"
                        ].iloc[-1],
                        self.initial_capital,
                    )
                )

            else:

                final_capital = (
                    self.initial_capital
                )

        else:

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
        )


        # ====================================================
        # 取引0回
        # ====================================================

        if (
            self.trades is None
            or self.trades.empty
        ):

            max_drawdown = (
                self._calculate_max_drawdown()
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

                "trades":
                    0,

                "trade_count":
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
                    max_drawdown,

                "average_holding_days":
                    None,

                "best_trade":
                    None,

                "worst_trade":
                    None,

                "total_commission":
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
                "Profit"
            ],
            errors="coerce",
        ).fillna(0.0)


        profit_rates = pd.to_numeric(
            self.trades[
                "Profit_Rate"
            ],
            errors="coerce",
        ).fillna(0.0)


        holding_days = pd.to_numeric(
            self.trades[
                "Holding_Days"
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
            self.trades
        )


        if trade_count > 0:

            win_rate = (
                wins
                / trade_count
            )

        else:

            win_rate = None


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

        if (
            "Total_Commission"
            in self.trades.columns
        ):

            total_commission = float(
                pd.to_numeric(
                    self.trades[
                        "Total_Commission"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

        else:

            total_commission = 0.0


        # ====================================================
        # 最大ドローダウン
        # ====================================================

        max_drawdown = (
            self._calculate_max_drawdown()
        )


        # ====================================================
        # 結果
        # ====================================================

        return {
            "initial_capital":
                self.initial_capital,

            "final_capital":
                final_capital,

            "total_profit":
                total_profit,

            "total_return":
                total_return,

            "trades":
                trade_count,

            "trade_count":
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
                    profit_rates.mean()
                ),

            "gross_profit":
                gross_profit,

            "gross_loss":
                gross_loss,

            "profit_factor":
                profit_factor,

            "max_drawdown":
                max_drawdown,

            "average_holding_days":
                float(
                    holding_days.mean()
                ),

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
    # 最大ドローダウン
    # ========================================================

    def _calculate_max_drawdown(
        self,
    ):

        if (
            self.equity_curve is None
            or self.equity_curve.empty
            or "Total_Equity"
            not in self.equity_curve.columns
        ):

            return 0.0


        equity = (
            pd.to_numeric(
                self.equity_curve[
                    "Total_Equity"
                ],
                errors="coerce",
            )
            .dropna()
        )


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
    # 結果取得
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


    trades, equity_curve, metrics = (
        engine.run(
            stock_data=
                stock_data,

            ai_data=
                ai_data,

            walk_results=
                walk_results,
        )
    )


    return (
        trades,
        equity_curve,
        metrics,
    )
