# ============================================================
# アドバンテスト AI売買システム
# backtest/report.py
#
# 売買バックテスト・レポート
#
# 目的
# ・ウォークフォワードAI予測を売買シグナルとして使用
# ・翌営業日の寄り付きで購入
# ・100株単位
# ・売買コスト / スリッページを考慮
# ・資産曲線を作成
# ・Buy & Holdと比較
#
# 現段階では
# 実際の注文は一切行わない
# ============================================================


import numpy as np
import pandas as pd


# ============================================================
# バックテストレポート
# ============================================================

class BacktestReport:

    """
    AIウォークフォワード予測から
    仮想売買の成績を計算するクラス
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        initial_capital=1_000_000,
        lot_size=100,
        entry_threshold=0.60,
        commission_rate=0.001,
        slippage_rate=0.001
    ):

        """
        initial_capital
            初期資金

        lot_size
            1回の最低売買株数

        entry_threshold
            AI上昇確率がこの値以上ならエントリー

        commission_rate
            売買1回あたりの手数料率

            例:
            0.001 = 0.1%

        slippage_rate
            想定価格から不利な方向にずれる割合

            例:
            0.001 = 0.1%
        """

        self.initial_capital = float(
            initial_capital
        )

        self.lot_size = int(
            lot_size
        )

        self.entry_threshold = float(
            entry_threshold
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


        if self.lot_size <= 0:

            raise ValueError(
                "lot_sizeは1以上にしてください。"
            )


        if not 0.0 < self.entry_threshold < 1.0:

            raise ValueError(
                "entry_thresholdは0〜1の間にしてください。"
            )


        if self.commission_rate < 0:

            raise ValueError(
                "commission_rateは0以上にしてください。"
            )


        if self.slippage_rate < 0:

            raise ValueError(
                "slippage_rateは0以上にしてください。"
            )


        self.trades = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.metrics = {}

        self.buy_hold = pd.DataFrame()


    # ========================================================
    # 株価データ準備
    # ========================================================

    @staticmethod
    def prepare_price_data(
        stock_data
    ):

        """
        株価データをバックテスト用に整理する。
        """

        if stock_data is None or stock_data.empty:

            raise ValueError(
                "株価データがありません。"
            )


        required_columns = [
            "Open",
            "Close"
        ]


        missing_columns = [

            column

            for column in required_columns

            if column not in stock_data.columns
        ]


        if missing_columns:

            raise ValueError(

                "バックテストに必要な株価列がありません："

                + ", ".join(
                    missing_columns
                )
            )


        df = stock_data.copy()


        # ----------------------------------------------------
        # 日付
        # ----------------------------------------------------

        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )


        df = df[
            ~df.index.isna()
        ].copy()


        try:

            if df.index.tz is not None:

                df.index = (
                    df.index
                    .tz_localize(None)
                )

        except AttributeError:

            pass


        df.index = (
            df.index.normalize()
        )


        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]


        df.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # 数値化
        # ----------------------------------------------------

        for column in required_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )


        df.dropna(
            subset=required_columns,
            inplace=True
        )


        return df


    # ========================================================
    # ウォークフォワード結果準備
    # ========================================================

    @staticmethod
    def prepare_prediction_data(
        walk_results
    ):

        """
        ウォークフォワード結果を整理する。
        """

        if (
            walk_results is None
            or walk_results.empty
        ):

            raise ValueError(
                "ウォークフォワード結果がありません。"
            )


        required_columns = [
            "Probability_Up"
        ]


        missing_columns = [

            column

            for column in required_columns

            if column not in walk_results.columns
        ]


        if missing_columns:

            raise ValueError(

                "ウォークフォワード結果に"
                "必要な列がありません："

                + ", ".join(
                    missing_columns
                )
            )


        df = walk_results.copy()


        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )


        df = df[
            ~df.index.isna()
        ].copy()


        try:

            if df.index.tz is not None:

                df.index = (
                    df.index
                    .tz_localize(None)
                )

        except AttributeError:

            pass


        df.index = (
            df.index.normalize()
        )


        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]


        df.sort_index(
            inplace=True
        )


        df["Probability_Up"] = (
            pd.to_numeric(
                df["Probability_Up"],
                errors="coerce"
            )
        )


        df.dropna(
            subset=[
                "Probability_Up"
            ],
            inplace=True
        )


        return df


    # ========================================================
    # 次の営業日取得
    # ========================================================

    @staticmethod
    def get_next_trading_date(
        price_index,
        signal_date
    ):

        """
        シグナル日の次に存在する
        株価データの日付を取得する。

        カレンダー上の翌日ではなく
        次の実際の取引日を使用する。
        """

        future_dates = price_index[
            price_index > signal_date
        ]


        if len(future_dates) == 0:

            return None


        return future_dates[0]


    # ========================================================
    # 約定価格
    # ========================================================

    def get_buy_price(
        self,
        open_price
    ):

        """
        買いはスリッページ分だけ
        不利な高い価格で約定したと仮定。
        """

        return float(
            open_price
            * (
                1.0
                + self.slippage_rate
            )
        )


    def get_sell_price(
        self,
        close_price
    ):

        """
        売りはスリッページ分だけ
        不利な安い価格で約定したと仮定。
        """

        return float(
            close_price
            * (
                1.0
                - self.slippage_rate
            )
        )


    # ========================================================
    # 売買バックテスト
    # ========================================================

    def run(
        self,
        stock_data,
        walk_results
    ):

        """
        AI上昇確率がentry_threshold以上なら

        シグナル日
            ↓
        次の営業日の寄り付きで購入
            ↓
        同日の終値で売却

        と仮定して検証する。

        これは日中1日保有型の初期戦略。
        """

        price_data = (
            self.prepare_price_data(
                stock_data
            )
        )


        prediction_data = (
            self.prepare_prediction_data(
                walk_results
            )
        )


        cash = float(
            self.initial_capital
        )


        trade_rows = []

        equity_rows = []


        # ----------------------------------------------------
        # バックテスト開始時点
        # ----------------------------------------------------

        if prediction_data.empty:

            raise ValueError(
                "予測データがありません。"
            )


        # ====================================================
        # シグナルを1日ずつ確認
        # ====================================================

        for (
            signal_date,
            prediction_row
        ) in prediction_data.iterrows():


            probability_up = float(
                prediction_row[
                    "Probability_Up"
                ]
            )


            # ------------------------------------------------
            # AI確率が基準未満なら取引しない
            # ------------------------------------------------

            if (
                probability_up
                < self.entry_threshold
            ):

                continue


            # ------------------------------------------------
            # 次の営業日
            # ------------------------------------------------

            trade_date = (
                self.get_next_trading_date(
                    price_data.index,
                    signal_date
                )
            )


            if trade_date is None:

                continue


            # ------------------------------------------------
            # 翌営業日の始値 / 終値
            # ------------------------------------------------

            open_price = float(
                price_data.loc[
                    trade_date,
                    "Open"
                ]
            )


            close_price = float(
                price_data.loc[
                    trade_date,
                    "Close"
                ]
            )


            if (
                open_price <= 0
                or close_price <= 0
            ):

                continue


            # ------------------------------------------------
            # スリッページ込み約定価格
            # ------------------------------------------------

            buy_price = (
                self.get_buy_price(
                    open_price
                )
            )


            sell_price = (
                self.get_sell_price(
                    close_price
                )
            )


            # ------------------------------------------------
            # 1単元の必要資金
            # ------------------------------------------------

            one_lot_value = (
                buy_price
                * self.lot_size
            )


            one_lot_buy_commission = (
                one_lot_value
                * self.commission_rate
            )


            one_lot_total_cost = (
                one_lot_value
                + one_lot_buy_commission
            )


            # ------------------------------------------------
            # 購入可能な単元数
            #
            # 現金の範囲内で最大購入
            # ------------------------------------------------

            lots = int(
                cash
                // one_lot_total_cost
            )


            if lots <= 0:

                continue


            shares = (
                lots
                * self.lot_size
            )


            # ------------------------------------------------
            # 買い
            # ------------------------------------------------

            buy_value = (
                buy_price
                * shares
            )


            buy_commission = (
                buy_value
                * self.commission_rate
            )


            total_buy_cost = (
                buy_value
                + buy_commission
            )


            # ------------------------------------------------
            # 売り
            # ------------------------------------------------

            sell_value = (
                sell_price
                * shares
            )


            sell_commission = (
                sell_value
                * self.commission_rate
            )


            net_sell_value = (
                sell_value
                - sell_commission
            )


            # ------------------------------------------------
            # 損益
            # ------------------------------------------------

            profit = (
                net_sell_value
                - total_buy_cost
            )


            profit_rate = (
                profit
                / total_buy_cost
            )


            # ------------------------------------------------
            # 現金更新
            #
            # 当日中に決済するので
            # 翌取引へ全額再利用可能
            # ------------------------------------------------

            capital_before = cash


            cash = (
                cash
                + profit
            )


            capital_after = cash


            # ------------------------------------------------
            # 総コスト
            # ------------------------------------------------

            total_commission = (
                buy_commission
                + sell_commission
            )


            slippage_cost = (

                (
                    buy_price
                    - open_price
                )
                * shares

                +

                (
                    close_price
                    - sell_price
                )
                * shares
            )


            # ------------------------------------------------
            # 勝敗
            # ------------------------------------------------

            win = (
                profit > 0
            )


            # ------------------------------------------------
            # 取引記録
            # ------------------------------------------------

            trade_rows.append(

                {

                    "Signal_Date":
                        signal_date,

                    "Trade_Date":
                        trade_date,

                    "Probability_Up":
                        probability_up,

                    "Open":
                        open_price,

                    "Close":
                        close_price,

                    "Buy_Price":
                        buy_price,

                    "Sell_Price":
                        sell_price,

                    "Shares":
                        int(
                            shares
                        ),

                    "Lots":
                        int(
                            lots
                        ),

                    "Buy_Value":
                        float(
                            buy_value
                        ),

                    "Sell_Value":
                        float(
                            sell_value
                        ),

                    "Buy_Commission":
                        float(
                            buy_commission
                        ),

                    "Sell_Commission":
                        float(
                            sell_commission
                        ),

                    "Total_Commission":
                        float(
                            total_commission
                        ),

                    "Slippage_Cost":
                        float(
                            slippage_cost
                        ),

                    "Profit":
                        float(
                            profit
                        ),

                    "Profit_Rate":
                        float(
                            profit_rate
                        ),

                    "Win":
                        bool(
                            win
                        ),

                    "Capital_Before":
                        float(
                            capital_before
                        ),

                    "Capital_After":
                        float(
                            capital_after
                        )
                }
            )


            # ------------------------------------------------
            # 資産曲線
            # ------------------------------------------------

            equity_rows.append(

                {

                    "Date":
                        trade_date,

                    "Equity":
                        float(
                            capital_after
                        ),

                    "Profit":
                        float(
                            profit
                        )
                }
            )


        # ====================================================
        # 取引結果
        # ====================================================

        self.trades = pd.DataFrame(
            trade_rows
        )


        if self.trades.empty:

            self.equity_curve = (
                pd.DataFrame()
            )

            self.metrics = {

                "initial_capital":
                    self.initial_capital,

                "final_capital":
                    self.initial_capital,

                "total_profit":
                    0.0,

                "total_return":
                    0.0,

                "trades":
                    0,

                "wins":
                    0,

                "losses":
                    0,

                "win_rate":
                    None,

                "profit_factor":
                    None,

                "max_drawdown":
                    0.0,

                "average_profit":
                    None,

                "average_profit_rate":
                    None,

                "average_win":
                    None,

                "average_loss":
                    None,

                "total_commission":
                    0.0,

                "total_slippage":
                    0.0,

                "entry_threshold":
                    self.entry_threshold
            }


            self.buy_hold = (
                self.calculate_buy_hold(
                    price_data,
                    prediction_data
                )
            )


            return (
                self.trades.copy(),
                self.equity_curve.copy(),
                self.metrics.copy()
            )


        # ----------------------------------------------------
        # 日付型
        # ----------------------------------------------------

        self.trades[
            "Signal_Date"
        ] = pd.to_datetime(
            self.trades[
                "Signal_Date"
            ]
        )


        self.trades[
            "Trade_Date"
        ] = pd.to_datetime(
            self.trades[
                "Trade_Date"
            ]
        )


        # ----------------------------------------------------
        # 資産曲線
        # ----------------------------------------------------

        self.equity_curve = (
            pd.DataFrame(
                equity_rows
            )
        )


        self.equity_curve[
            "Date"
        ] = pd.to_datetime(
            self.equity_curve[
                "Date"
            ]
        )


        self.equity_curve.set_index(
            "Date",
            inplace=True
        )


        self.equity_curve.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # 初期資産を追加
        # ----------------------------------------------------

        first_trade_date = (
            self.equity_curve.index[0]
        )


        initial_row = pd.DataFrame(

            {
                "Equity":
                    [
                        self.initial_capital
                    ],

                "Profit":
                    [
                        0.0
                    ]
            },

            index=[
                first_trade_date
                - pd.Timedelta(
                    days=1
                )
            ]
        )


        self.equity_curve = (
            pd.concat(
                [
                    initial_row,
                    self.equity_curve
                ]
            )
            .sort_index()
        )


        # ----------------------------------------------------
        # 累積リターン
        # ----------------------------------------------------

        self.equity_curve[
            "Cumulative_Return"
        ] = (
            self.equity_curve[
                "Equity"
            ]
            / self.initial_capital
            - 1.0
        )


        # ----------------------------------------------------
        # ドローダウン
        # ----------------------------------------------------

        rolling_peak = (
            self.equity_curve[
                "Equity"
            ]
            .cummax()
        )


        self.equity_curve[
            "Drawdown"
        ] = (
            self.equity_curve[
                "Equity"
            ]
            / rolling_peak
            - 1.0
        )


        # ====================================================
        # Buy & Hold
        # ====================================================

        self.buy_hold = (
            self.calculate_buy_hold(
                price_data,
                prediction_data
            )
        )


        # ====================================================
        # 評価指標
        # ====================================================

        self.metrics = (
            self.calculate_metrics()
        )


        return (
            self.trades.copy(),
            self.equity_curve.copy(),
            self.metrics.copy()
        )


    # ========================================================
    # 評価指標
    # ========================================================

    def calculate_metrics(
        self
    ):

        """
        売買成績を集計する。
        """

        if self.trades.empty:

            return {}


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


        winning_trades = (
            self.trades[
                self.trades[
                    "Profit"
                ]
                > 0
            ]
        )


        losing_trades = (
            self.trades[
                self.trades[
                    "Profit"
                ]
                < 0
            ]
        )


        # ----------------------------------------------------
        # 基本
        # ----------------------------------------------------

        final_capital = float(
            self.trades.iloc[-1][
                "Capital_After"
            ]
        )


        total_profit = (
            final_capital
            - self.initial_capital
        )


        total_return = (
            final_capital
            / self.initial_capital
            - 1.0
        )


        trade_count = int(
            len(
                self.trades
            )
        )


        wins = int(
            len(
                winning_trades
            )
        )


        losses = int(
            len(
                losing_trades
            )
        )


        flat_trades = (
            trade_count
            - wins
            - losses
        )


        # ----------------------------------------------------
        # 勝率
        # ----------------------------------------------------

        if trade_count > 0:

            win_rate = (
                wins
                / trade_count
            )

        else:

            win_rate = None


        # ----------------------------------------------------
        # 平均損益
        # ----------------------------------------------------

        average_profit = float(
            profits.mean()
        )


        average_profit_rate = float(
            profit_rates.mean()
        )


        # ----------------------------------------------------
        # 平均利益
        # ----------------------------------------------------

        if wins > 0:

            average_win = float(
                winning_trades[
                    "Profit"
                ]
                .mean()
            )

        else:

            average_win = None


        # ----------------------------------------------------
        # 平均損失
        # ----------------------------------------------------

        if losses > 0:

            average_loss = float(
                losing_trades[
                    "Profit"
                ]
                .mean()
            )

        else:

            average_loss = None


        # ----------------------------------------------------
        # Gross Profit / Loss
        # ----------------------------------------------------

        gross_profit = float(

            winning_trades[
                "Profit"
            ].sum()

            if wins > 0

            else 0.0
        )


        gross_loss = float(

            abs(
                losing_trades[
                    "Profit"
                ].sum()
            )

            if losses > 0

            else 0.0
        )


        # ----------------------------------------------------
        # Profit Factor
        # ----------------------------------------------------

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        elif gross_profit > 0:

            profit_factor = None

        else:

            profit_factor = 0.0


        # ----------------------------------------------------
        # 最大ドローダウン
        # ----------------------------------------------------

        if (
            self.equity_curve is not None
            and not self.equity_curve.empty
        ):

            max_drawdown = float(
                self.equity_curve[
                    "Drawdown"
                ]
                .min()
            )

        else:

            max_drawdown = 0.0


        # ----------------------------------------------------
        # 最大利益 / 最大損失
        # ----------------------------------------------------

        best_trade = float(
            profits.max()
        )


        worst_trade = float(
            profits.min()
        )


        # ----------------------------------------------------
        # コスト
        # ----------------------------------------------------

        total_commission = float(
            self.trades[
                "Total_Commission"
            ]
            .sum()
        )


        total_slippage = float(
            self.trades[
                "Slippage_Cost"
            ]
            .sum()
        )


        # ----------------------------------------------------
        # Buy & Hold
        # ----------------------------------------------------

        buy_hold_return = None


        if (
            self.buy_hold is not None
            and not self.buy_hold.empty
        ):

            buy_hold_return = float(
                self.buy_hold.iloc[-1][
                    "BuyHold_Return"
                ]
            )


        # ----------------------------------------------------
        # AI戦略とBuy & Holdの差
        # ----------------------------------------------------

        if buy_hold_return is not None:

            excess_return = (
                total_return
                - buy_hold_return
            )

        else:

            excess_return = None


        # ----------------------------------------------------
        # 結果
        # ----------------------------------------------------

        metrics = {

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

            "trades":
                trade_count,

            "wins":
                wins,

            "losses":
                losses,

            "flat_trades":
                int(
                    flat_trades
                ),

            "win_rate":
                (
                    float(
                        win_rate
                    )
                    if win_rate is not None
                    else None
                ),

            "average_profit":
                average_profit,

            "average_profit_rate":
                average_profit_rate,

            "average_win":
                average_win,

            "average_loss":
                average_loss,

            "gross_profit":
                gross_profit,

            "gross_loss":
                gross_loss,

            "profit_factor":
                (
                    float(
                        profit_factor
                    )
                    if profit_factor is not None
                    else None
                ),

            "best_trade":
                best_trade,

            "worst_trade":
                worst_trade,

            "max_drawdown":
                max_drawdown,

            "total_commission":
                total_commission,

            "total_slippage":
                total_slippage,

            "buy_hold_return":
                buy_hold_return,

            "excess_return":
                (
                    float(
                        excess_return
                    )
                    if excess_return is not None
                    else None
                ),

            "entry_threshold":
                float(
                    self.entry_threshold
                ),

            "lot_size":
                int(
                    self.lot_size
                ),

            "commission_rate":
                float(
                    self.commission_rate
                ),

            "slippage_rate":
                float(
                    self.slippage_rate
                )
        }


        return metrics


    # ========================================================
    # Buy & Hold
    # ========================================================

    def calculate_buy_hold(
        self,
        price_data,
        prediction_data
    ):

        """
        ウォークフォワード検証期間に対応した
        Buy & Hold参考値を計算する。

        最初の予測日の次の営業日始値で購入し、
        最後の利用可能な終値まで保有する。

        比較用の参考値。
        """

        if (
            price_data is None
            or price_data.empty
            or prediction_data is None
            or prediction_data.empty
        ):

            return pd.DataFrame()


        first_signal_date = (
            prediction_data.index[0]
        )


        last_signal_date = (
            prediction_data.index[-1]
        )


        start_date = (
            self.get_next_trading_date(
                price_data.index,
                first_signal_date
            )
        )


        if start_date is None:

            return pd.DataFrame()


        # ----------------------------------------------------
        # 最後のシグナルより後の営業日があれば
        # そこまで含める
        # ----------------------------------------------------

        possible_end_dates = (
            price_data.index[
                price_data.index
                > last_signal_date
            ]
        )


        if len(possible_end_dates) > 0:

            end_date = (
                possible_end_dates[0]
            )

        else:

            end_date = (
                price_data.index[-1]
            )


        comparison_data = (
            price_data.loc[
                (
                    price_data.index
                    >= start_date
                )
                &
                (
                    price_data.index
                    <= end_date
                )
            ]
            .copy()
        )


        if comparison_data.empty:

            return pd.DataFrame()


        # ----------------------------------------------------
        # 開始価格
        # ----------------------------------------------------

        start_open = float(
            comparison_data.iloc[0][
                "Open"
            ]
        )


        if start_open <= 0:

            return pd.DataFrame()


        # ----------------------------------------------------
        # Buy & Holdリターン
        # ----------------------------------------------------

        comparison_data[
            "BuyHold_Return"
        ] = (
            comparison_data[
                "Close"
            ]
            / start_open
            - 1.0
        )


        comparison_data[
            "BuyHold_Equity"
        ] = (
            self.initial_capital
            * (
                1.0
                + comparison_data[
                    "BuyHold_Return"
                ]
            )
        )


        return comparison_data[
            [
                "Close",
                "BuyHold_Return",
                "BuyHold_Equity"
            ]
        ].copy()


    # ========================================================
    # 比較用資産曲線
    # ========================================================

    def get_comparison_curve(
        self
    ):

        """
        AI戦略とBuy & Holdを
        同じグラフに表示するためのデータを作る。
        """

        if (
            self.equity_curve is None
            or self.equity_curve.empty
        ):

            return pd.DataFrame()


        comparison = (
            self.equity_curve[
                [
                    "Equity"
                ]
            ]
            .copy()
        )


        comparison.rename(
            columns={
                "Equity":
                    "AI_Strategy"
            },
            inplace=True
        )


        if (
            self.buy_hold is not None
            and not self.buy_hold.empty
        ):

            comparison = (
                comparison.join(
                    self.buy_hold[
                        [
                            "BuyHold_Equity"
                        ]
                    ],
                    how="outer"
                )
            )


            comparison.sort_index(
                inplace=True
            )


            comparison[
                "AI_Strategy"
            ] = (
                comparison[
                    "AI_Strategy"
                ]
                .ffill()
                .fillna(
                    self.initial_capital
                )
            )


            comparison[
                "BuyHold_Equity"
            ] = (
                comparison[
                    "BuyHold_Equity"
                ]
                .ffill()
            )


        return comparison


    # ========================================================
    # 取引一覧
    # ========================================================

    def get_trades(
        self
    ):

        return (
            self.trades.copy()
        )


    # ========================================================
    # 資産曲線
    # ========================================================

    def get_equity_curve(
        self
    ):

        return (
            self.equity_curve.copy()
        )


    # ========================================================
    # Buy & Hold
    # ========================================================

    def get_buy_hold(
        self
    ):

        return (
            self.buy_hold.copy()
        )


    # ========================================================
    # 指標
    # ========================================================

    def get_metrics(
        self
    ):

        return (
            self.metrics.copy()
        )


# ============================================================
# 簡単実行関数
# ============================================================

def create_backtest_report(
    stock_data,
    walk_results,
    initial_capital=1_000_000,
    lot_size=100,
    entry_threshold=0.60,
    commission_rate=0.001,
    slippage_rate=0.001
):

    """
    バックテストを一度に実行する便利関数。
    """

    report = BacktestReport(

        initial_capital=
            initial_capital,

        lot_size=
            lot_size,

        entry_threshold=
            entry_threshold,

        commission_rate=
            commission_rate,

        slippage_rate=
            slippage_rate
    )


    (
        trades,
        equity_curve,
        metrics
    ) = report.run(

        stock_data=
            stock_data,

        walk_results=
            walk_results
    )


    return (
        report,
        trades,
        equity_curve,
        metrics
    )
