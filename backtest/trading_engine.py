# ============================================================
# アドバンテスト AI売買システム
# backtest/trading_engine.py
#
# 本格戦略バックテストエンジン
#
# 流れ
# 1. ウォークフォワードAI予測を読む
# 2. EntryStrategyでBUY判定
# 3. 翌営業日の始値で購入
# 4. RiskManagerで購入株数を決定
# 5. ポジションを保有
# 6. ExitStrategyで毎日SELL/HOLD判定
# 7. SELL条件成立で売却
# 8. 資金を更新
# 9. 次のエントリーを待つ
#
# 実際の注文は行わない
# ============================================================


import numpy as np
import pandas as pd


from strategy.entry import EntryStrategy
from strategy.exit import ExitStrategy
from strategy.risk import RiskManager


# ============================================================
# 本格売買バックテスト
# ============================================================

class TradingBacktestEngine:

    """
    EntryStrategy
    ExitStrategy
    RiskManager

    を統合した売買バックテスト。
    """


    # ========================================================
    # 初期化
    # ========================================================

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
        slippage_rate=0.001
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


        if self.initial_capital <= 0:

            raise ValueError(
                "initial_capitalは0より大きくしてください。"
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

            rsi_min=
                40.0,

            rsi_max=
                70.0,

            volume_ratio_min=
                1.0
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

            rsi_overbought=
                75.0
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
                slippage_rate
        )


        # ====================================================
        # 結果保存
        # ====================================================

        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.metrics = {}

        self.skipped_entries = []


    # ========================================================
    # 日付インデックス整理
    # ========================================================

    @staticmethod
    def _prepare_index(
        data
    ):

        """
        DatetimeIndexへ統一する。
        """

        df = data.copy()


        df.index = pd.to_datetime(
            df.index
        )


        # timezone除去
        try:

            if df.index.tz is not None:

                df.index = (
                    df.index.tz_localize(
                        None
                    )
                )

        except AttributeError:

            pass


        df = (
            df[
                ~df.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )


        return df


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        stock_data,
        ai_data,
        walk_results
    ):

        """
        株価・AI特徴量・ウォークフォワード結果を
        バックテスト用に整理する。
        """


        if (
            stock_data is None
            or stock_data.empty
        ):

            raise ValueError(
                "stock_dataがありません。"
            )


        if (
            ai_data is None
            or ai_data.empty
        ):

            raise ValueError(
                "ai_dataがありません。"
            )


        if (
            walk_results is None
            or walk_results.empty
        ):

            raise ValueError(
                "walk_resultsがありません。"
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


        # ----------------------------------------------------
        # 必須株価列
        # ----------------------------------------------------

        required_stock_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]


        missing = [

            column

            for column
            in required_stock_columns

            if column
            not in stock.columns
        ]


        if missing:

            raise ValueError(
                "stock_dataに必要な列がありません: "
                + ", ".join(
                    missing
                )
            )


        # ----------------------------------------------------
        # AI確率
        # ----------------------------------------------------

        if (
            "Probability_Up"
            not in predictions.columns
        ):

            raise ValueError(
                "walk_resultsにProbability_Upがありません。"
            )


        return (
            stock,
            features,
            predictions
        )


    # ========================================================
    # 次の営業日
    # ========================================================

    @staticmethod
    def get_next_trading_date(
        stock_data,
        signal_date
    ):

        """
        シグナル日の次の株価データ日を取得する。
        """

        future_dates = (
            stock_data.index[
                stock_data.index
                > signal_date
            ]
        )


        if len(
            future_dates
        ) == 0:

            return None


        return future_dates[0]


    # ========================================================
    # AI確率取得
    # ========================================================

    @staticmethod
    def get_probability(
        predictions,
        date
    ):

        """
        指定日のウォークフォワードAI確率を取得。
        """

        if (
            date
            not in predictions.index
        ):

            return None


        value = predictions.loc[
            date,
            "Probability_Up"
        ]


        if isinstance(
            value,
            pd.Series
        ):

            value = value.iloc[-1]


        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None


        if not np.isfinite(
            value
        ):

            return None


        return value


    # ========================================================
    # 売却価格
    # ========================================================

    def calculate_sell_price(
        self,
        market_price
    ):

        """
        売却時のスリッページを反映。
        """

        market_price = float(
            market_price
        )


        return (
            market_price
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
        shares
    ):

        return (
            float(
                sell_price
            )
            * int(
                shares
            )
            * self.commission_rate
        )


    # ========================================================
    # バックテスト実行
    # ========================================================

    def run(
        self,
        stock_data,
        ai_data,
        walk_results
    ):

        """
        本格戦略バックテストを実行する。
        """


        (
            stock,
            features,
            predictions
        ) = self.prepare_data(

            stock_data=
                stock_data,

            ai_data=
                ai_data,

            walk_results=
                walk_results
        )


        # ====================================================
        # 初期状態
        # ====================================================

        cash = float(
            self.initial_capital
        )


        position = None

        trade_records = []

        equity_records = []

        skipped_entries = []


        # ====================================================
        # バックテスト対象期間
        # ====================================================

        start_date = (
            predictions.index.min()
        )


        trading_dates = (
            stock.index[
                stock.index
                >= start_date
            ]
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
                pd.DataFrame
            ):

                stock_row = (
                    stock_row.iloc[-1]
                )


            close_price = float(
                stock_row[
                    "Close"
                ]
            )


            # =================================================
            # ① ポジション保有中
            # =================================================

            if position is not None:

                position[
                    "holding_days"
                ] += 1


                # ---------------------------------------------
                # 特徴量がある日のみ
                # AI・テクニカル売却判定を実施
                # ---------------------------------------------

                if (
                    current_date
                    in features.index
                ):

                    feature_row = (
                        features.loc[
                            current_date
                        ]
                    )


                    if isinstance(
                        feature_row,
                        pd.DataFrame
                    ):

                        feature_row = (
                            feature_row.iloc[-1]
                        )


                    probability_up = (
                        self.get_probability(
                            predictions,
                            current_date
                        )
                    )


                    exit_result = (
                        self.exit_strategy.evaluate(
                            row=feature_row,
                            entry_price=position[
                                "entry_price"
                            ],
                            current_price=close_price,
                            probability_up=probability_up,
                            holding_days=position[
                                "holding_days"
                            ]
                        )
                    )


                    # =========================================
                    # SELL
                    # =========================================

                    if (
                        exit_result[
                            "action"
                        ]
                        == "SELL"
                    ):

                        sell_price = (
                            self.calculate_sell_price(
                                close_price
                            )
                        )


                        sell_commission = (
                            self.calculate_sell_commission(
                                sell_price=
                                    sell_price,

                                shares=
                                    position[
                                        "shares"
                                    ]
                            )
                        )


                        sale_value = (
                            sell_price
                            * position[
                                "shares"
                            ]
                        )


                        cash += (
                            sale_value
                            - sell_commission
                        )


                        # -------------------------------------
                        # 実現損益
                        # -------------------------------------

                        total_buy_cost = (
                            position[
                                "required_cash"
                            ]
                        )


                        net_sale_value = (
                            sale_value
                            - sell_commission
                        )


                        profit = (
                            net_sale_value
                            - total_buy_cost
                        )


                        if (
                            total_buy_cost
                            > 0
                        ):

                            profit_rate = (
                                profit
                                / total_buy_cost
                            )

                        else:

                            profit_rate = 0.0


                        # -------------------------------------
                        # 取引保存
                        # -------------------------------------

                        trade_records.append(
                            {

                                "Signal_Date":
                                    position[
                                        "signal_date"
                                    ],

                                "Entry_Date":
                                    position[
                                        "entry_date"
                                    ],

                                "Exit_Date":
                                    current_date,

                                "Entry_Probability":
                                    position[
                                        "entry_probability"
                                    ],

                                "Entry_Score":
                                    position[
                                        "entry_score"
                                    ],

                                "Entry_Price":
                                    position[
                                        "entry_price"
                                    ],

                                "Exit_Price":
                                    sell_price,

                                "Shares":
                                    position[
                                        "shares"
                                    ],

                                "Holding_Days":
                                    position[
                                        "holding_days"
                                    ],

                                "Buy_Value":
                                    position[
                                        "purchase_value"
                                    ],

                                "Buy_Commission":
                                    position[
                                        "buy_commission"
                                    ],

                                "Sell_Commission":
                                    sell_commission,

                                "Profit":
                                    profit,

                                "Profit_Rate":
                                    profit_rate,

                                "Exit_Type":
                                    exit_result[
                                        "exit_type"
                                    ],

                                "Exit_Reason":
                                    exit_result[
                                        "summary"
                                    ],

                                "Exit_Score":
                                    exit_result[
                                        "exit_score"
                                    ],

                                "Capital_After":
                                    cash
                            }
                        )


                        position = None


            # =================================================
            # ② ポジションなし
            # =================================================

            if position is None:

                # ---------------------------------------------
                # 当日のAI予測があるか
                # ---------------------------------------------

                probability_up = (
                    self.get_probability(
                        predictions,
                        current_date
                    )
                )


                if (
                    probability_up
                    is not None
                    and current_date
                    in features.index
                ):

                    feature_row = (
                        features.loc[
                            current_date
                        ]
                    )


                    if isinstance(
                        feature_row,
                        pd.DataFrame
                    ):

                        feature_row = (
                            feature_row.iloc[-1]
                        )


                    # =========================================
                    # EntryStrategy
                    # =========================================

                    entry_result = (
                        self.entry_strategy.evaluate(
                            row=feature_row,
                            probability_up=
                                probability_up
                        )
                    )


                    # =========================================
                    # BUYシグナル
                    # =========================================

                    if (
                        entry_result[
                            "action"
                        ]
                        == "BUY"
                    ):

                        next_date = (
                            self.get_next_trading_date(
                                stock_data=stock,
                                signal_date=current_date
                            )
                        )


                        if next_date is not None:

                            next_row = (
                                stock.loc[
                                    next_date
                                ]
                            )


                            if isinstance(
                                next_row,
                                pd.DataFrame
                            ):

                                next_row = (
                                    next_row.iloc[-1]
                                )


                            next_open = float(
                                next_row[
                                    "Open"
                                ]
                            )


                            # =================================
                            # RiskManager
                            # =================================

                            risk_result = (
                                self.risk_manager.evaluate_trade(
                                    capital=cash,
                                    market_price=next_open
                                )
                            )


                            if (
                                risk_result[
                                    "can_trade"
                                ]
                            ):

                                # -----------------------------
                                # BUY
                                # -----------------------------

                                required_cash = (
                                    risk_result[
                                        "required_cash"
                                    ]
                                )


                                if (
                                    required_cash
                                    <= cash
                                ):

                                    cash -= (
                                        required_cash
                                    )


                                    position = {

                                        "signal_date":
                                            current_date,

                                        "entry_date":
                                            next_date,

                                        "entry_probability":
                                            probability_up,

                                        "entry_score":
                                            entry_result[
                                                "score"
                                            ],

                                        "entry_price":
                                            risk_result[
                                                "entry_price"
                                            ],

                                        "stop_price":
                                            risk_result[
                                                "stop_price"
                                            ],

                                        "take_profit_price":
                                            risk_result[
                                                "take_profit_price"
                                            ],

                                        "shares":
                                            risk_result[
                                                "shares"
                                            ],

                                        "purchase_value":
                                            risk_result[
                                                "purchase_value"
                                            ],

                                        "buy_commission":
                                            risk_result[
                                                "buy_commission"
                                            ],

                                        "required_cash":
                                            required_cash,

                                        "holding_days":
                                            0
                                    }


                            else:

                                skipped_entries.append(
                                    {

                                        "Date":
                                            current_date,

                                        "Probability_Up":
                                            probability_up,

                                        "Entry_Score":
                                            entry_result[
                                                "score"
                                            ],

                                        "Reason":
                                            risk_result[
                                                "reason"
                                            ]
                                    }
                                )


            # =================================================
            # ③ 日次資産評価
            # =================================================

            if position is None:

                market_value = 0.0

            else:

                # ------------------------------------------------
                # エントリー日は未来日になるため、
                # シグナル日にはまだ保有していない扱いにする。
                # ------------------------------------------------

                if (
                    current_date
                    < position[
                        "entry_date"
                    ]
                ):

                    market_value = 0.0

                else:

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

                    "Position_Value":
                        market_value,

                    "Total_Equity":
                        total_equity
                }
            )


        # ====================================================
        # 最終日にポジションが残っている場合
        # ====================================================

        if position is not None:

            final_date = (
                trading_dates[-1]
            )


            final_row = (
                stock.loc[
                    final_date
                ]
            )


            if isinstance(
                final_row,
                pd.DataFrame
            ):

                final_row = (
                    final_row.iloc[-1]
                )


            final_close = float(
                final_row[
                    "Close"
                ]
            )


            sell_price = (
                self.calculate_sell_price(
                    final_close
                )
            )


            sell_commission = (
                self.calculate_sell_commission(
                    sell_price=
                        sell_price,

                    shares=
                        position[
                            "shares"
                        ]
                )
            )


            sale_value = (
                sell_price
                * position[
                    "shares"
                ]
            )


            cash += (
                sale_value
                - sell_commission
            )


            total_buy_cost = (
                position[
                    "required_cash"
                ]
            )


            net_sale_value = (
                sale_value
                - sell_commission
            )


            profit = (
                net_sale_value
                - total_buy_cost
            )


            profit_rate = (
                profit
                / total_buy_cost
                if total_buy_cost > 0
                else 0.0
            )


            trade_records.append(
                {

                    "Signal_Date":
                        position[
                            "signal_date"
                        ],

                    "Entry_Date":
                        position[
                            "entry_date"
                        ],

                    "Exit_Date":
                        final_date,

                    "Entry_Probability":
                        position[
                            "entry_probability"
                        ],

                    "Entry_Score":
                        position[
                            "entry_score"
                        ],

                    "Entry_Price":
                        position[
                            "entry_price"
                        ],

                    "Exit_Price":
                        sell_price,

                    "Shares":
                        position[
                            "shares"
                        ],

                    "Holding_Days":
                        position[
                            "holding_days"
                        ],

                    "Buy_Value":
                        position[
                            "purchase_value"
                        ],

                    "Buy_Commission":
                        position[
                            "buy_commission"
                        ],

                    "Sell_Commission":
                        sell_commission,

                    "Profit":
                        profit,

                    "Profit_Rate":
                        profit_rate,

                    "Exit_Type":
                        "END_OF_TEST",

                    "Exit_Reason":
                        "バックテスト最終日のため決済",

                    "Exit_Score":
                        0,

                    "Capital_After":
                        cash
                }
            )


            position = None


            # ------------------------------------------------
            # 最終資産を更新
            # ------------------------------------------------

            if len(
                equity_records
            ) > 0:

                equity_records[-1][
                    "Cash"
                ] = cash

                equity_records[-1][
                    "Position_Value"
                ] = 0.0

                equity_records[-1][
                    "Total_Equity"
                ] = cash


        # ====================================================
        # DataFrame化
        # ====================================================

        self.trades = pd.DataFrame(
            trade_records
        )


        self.equity_curve = pd.DataFrame(
            equity_records
        )


        if not self.equity_curve.empty:

            self.equity_curve.set_index(
                "Date",
                inplace=True
            )


        self.skipped_entries = (
            skipped_entries
        )


        # ====================================================
        # 成績計算
        # ====================================================

        self.metrics = (
            self.calculate_metrics()
        )


        return (
            self.trades,
            self.equity_curve,
            self.metrics
        )


    # ========================================================
    # 最大ドローダウン
    # ========================================================

    def calculate_max_drawdown(
        self
    ):

        if (
            self.equity_curve is None
            or self.equity_curve.empty
        ):

            return 0.0


        equity = (
            self.equity_curve[
                "Total_Equity"
            ]
        )


        running_max = (
            equity.cummax()
        )


        drawdown = (
            equity
            / running_max
            - 1.0
        )


        return float(
            drawdown.min()
        )


    # ========================================================
    # 成績
    # ========================================================

    def calculate_metrics(
        self
    ):

        """
        バックテスト結果を集計する。
        """


        if (
            self.equity_curve is None
            or self.equity_curve.empty
        ):

            final_capital = (
                self.initial_capital
            )

        else:

            final_capital = float(
                self.equity_curve[
                    "Total_Equity"
                ].iloc[-1]
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
        # 取引なし
        # ====================================================

        if (
            self.trades is None
            or self.trades.empty
        ):

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

                "wins":
                    0,

                "losses":
                    0,

                "win_rate":
                    None,

                "average_profit":
                    None,

                "average_profit_rate":
                    None,

                "profit_factor":
                    None,

                "max_drawdown":
                    self.calculate_max_drawdown(),

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
                    )
            }


        # ====================================================
        # 通常集計
        # ====================================================

        profits = (
            self.trades[
                "Profit"
            ]
        )


        profit_rates = (
            self.trades[
                "Profit_Rate"
            ]
        )


        wins = (
            profits
            > 0
        )


        losses = (
            profits
            < 0
        )


        win_count = int(
            wins.sum()
        )


        loss_count = int(
            losses.sum()
        )


        trade_count = int(
            len(
                self.trades
            )
        )


        win_rate = (
            win_count
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


        total_commission = float(

            self.trades[
                "Buy_Commission"
            ].sum()

            +

            self.trades[
                "Sell_Commission"
            ].sum()
        )


        return {

            "initial_capital":
                float(
                    self.initial_capital
                ),

            "final_capital":
                float(
                    final_capital
                ),

            "total_profit":
                float(
                    total_profit
                ),

            "total_return":
                float(
                    total_return
                ),

            "trades":
                trade_count,

            "wins":
                win_count,

            "losses":
                loss_count,

            "win_rate":
                float(
                    win_rate
                ),

            "average_profit":
                float(
                    profits.mean()
                ),

            "average_profit_rate":
                float(
                    profit_rates.mean()
                ),

            "profit_factor":
                profit_factor,

            "max_drawdown":
                float(
                    self.calculate_max_drawdown()
                ),

            "average_holding_days":
                float(
                    self.trades[
                        "Holding_Days"
                    ].mean()
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
                )
        }


    # ========================================================
    # 取引履歴取得
    # ========================================================

    def get_trades(
        self
    ):

        return self.trades.copy()


    # ========================================================
    # 資産曲線取得
    # ========================================================

    def get_equity_curve(
        self
    ):

        return self.equity_curve.copy()


    # ========================================================
    # 成績取得
    # ========================================================

    def get_metrics(
        self
    ):

        return dict(
            self.metrics
        )


    # ========================================================
    # 資金不足などで見送ったシグナル
    # ========================================================

    def get_skipped_entries(
        self
    ):

        return pd.DataFrame(
            self.skipped_entries
        )


# ============================================================
# 簡単実行関数
# ============================================================

def run_trading_backtest(
    stock_data,
    ai_data,
    walk_results,
    initial_capital=1_000_000
):

    """
    本格戦略バックテストを
    一度に実行する便利関数。
    """

    engine = TradingBacktestEngine(
        initial_capital=
            initial_capital
    )


    return engine.run(

        stock_data=
            stock_data,

        ai_data=
            ai_data,

        walk_results=
            walk_results
    )
