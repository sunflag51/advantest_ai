# ============================================================
# アドバンテスト AI売買分析システム
# main.py
#
# ①～⑬ 統合完全版
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# データ
# ============================================================

from data.stock_data import (
    get_stock_data,
)

from data.market_data import (
    get_all_market_data,
    get_latest_market_values,
)


# ============================================================
# テクニカル
# ============================================================

from indicators.technical import (
    add_all_indicators,
)


# ============================================================
# AI
# ============================================================

from ai.features import (
    build_ai_features,
)

from ai.model import (
    StockPredictionModel,
)


# ============================================================
# バックテスト
# ============================================================

from backtest.engine import (
    WalkForwardBacktest,
)

from backtest.report import (
    BacktestReport,
)

from backtest.trading_engine import (
    TradingBacktestEngine,
)


# ============================================================
# 戦略
# ============================================================

from strategy.entry import (
    EntryStrategy,
)

from strategy.risk import (
    RiskManager,
)


# ============================================================
# 基本設定
# ============================================================

STOCK_CODE = "6857.T"

STOCK_NAME = "アドバンテスト"

INITIAL_CAPITAL = 1_000_000


# ============================================================
# Streamlit設定
# ============================================================

st.set_page_config(
    page_title=
        "アドバンテスト AI売買分析",

    page_icon=
        "📈",

    layout=
        "wide",
)


# ============================================================
# タイトル
# ============================================================

st.title(
    "📈 アドバンテスト AI売買分析システム"
)

st.caption(
    "株価・市場・テクニカル・AI・"
    "ウォークフォワード・売買戦略・"
    "リスク管理を統合して検証します。"
)


# ============================================================
# 安全なfloat変換
# ============================================================

def safe_float(
    value,
    default=None,
):

    try:

        if value is None:

            return default


        # DataFrame
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


        # Series
        if isinstance(
            value,
            pd.Series,
        ):

            numeric = (
                pd.to_numeric(
                    value,
                    errors="coerce",
                )
                .dropna()
            )

            if numeric.empty:

                return default

            value = numeric.iloc[-1]


        # numpy/list/tuple
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


# ============================================================
# Metric表示用
# ============================================================

def safe_metric_number(
    value,
    digits=2,
    suffix="",
    default="N/A",
):

    number = safe_float(
        value
    )


    if number is None:

        return default


    return (
        f"{number:,.{digits}f}"
        f"{suffix}"
    )


# ============================================================
# パーセント表示
# ============================================================

def format_percent(
    value,
    digits=2,
):

    number = safe_float(
        value
    )


    if number is None:

        return "N/A"


    return (
        f"{number * 100:.{digits}f}%"
    )


# ============================================================
# 実行ボタン
# ============================================================

run_analysis = st.button(
    "🚀 株価・市場・AI・戦略分析を実行",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 実行
# ============================================================

if run_analysis:

    try:

        # ====================================================
        # ① 株価取得
        # ====================================================

        st.header(
            "① 📊 アドバンテスト株価"
        )


        with st.spinner(
            "株価データを取得しています..."
        ):

            stock_data = get_stock_data(
                stock_code=
                    STOCK_CODE,

                period=
                    "5y",

                interval=
                    "1d",
            )


        if (
            stock_data is None
            or stock_data.empty
        ):

            st.error(
                "株価データを取得できませんでした。"
            )

            st.stop()


        latest_close = safe_float(
            stock_data[
                "Close"
            ].iloc[-1]
        )


        latest_date = (
            stock_data.index[-1]
        )


        col1, col2, col3 = st.columns(
            3
        )


        col1.metric(
            "銘柄",
            STOCK_NAME,
        )


        col2.metric(
            "最新株価",
            (
                f"{latest_close:,.0f} 円"

                if latest_close
                is not None

                else "取得なし"
            ),
        )


        col3.metric(
            "最新日",
            str(
                latest_date.date()
            ),
        )


        st.line_chart(
            stock_data[
                ["Close"]
            ]
        )


        st.success(
            "株価データ取得完了"
        )


        # ====================================================
        # ② 市場データ
        # ====================================================

        st.header(
            "② 🌏 市場データ"
        )


        with st.spinner(
            "市場データを取得しています..."
        ):

            market_data = (
                get_all_market_data(
                    period=
                        "5y",

                    interval=
                        "1d",
                )
            )


            latest_market = (
                get_latest_market_values(
                    period=
                        "5d",

                    interval=
                        "1d",
                )
            )


        market_names = [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円",
        ]


        market_columns = st.columns(
            4
        )


        for index, name in enumerate(
            market_names
        ):

            value = None


            if isinstance(
                latest_market,
                dict,
            ):

                value = (
                    latest_market.get(
                        name
                    )
                )


            display_value = safe_float(
                value
            )


            if display_value is None:

                market_columns[
                    index
                ].metric(
                    name,
                    "取得なし",
                )

            else:

                market_columns[
                    index
                ].metric(
                    name,
                    f"{display_value:,.2f}",
                )


        # ----------------------------------------------------
        # 市場比較チャート
        # ----------------------------------------------------

        normalized_market = (
            pd.DataFrame()
        )


        if isinstance(
            market_data,
            dict,
        ):

            for (
                market_name,
                df,
            ) in market_data.items():

                if (
                    isinstance(
                        df,
                        pd.DataFrame,
                    )
                    and not df.empty
                    and "Close"
                    in df.columns
                ):

                    close = (
                        pd.to_numeric(
                            df[
                                "Close"
                            ],
                            errors="coerce",
                        )
                        .dropna()
                    )


                    if not close.empty:

                        first_value = (
                            safe_float(
                                close.iloc[0]
                            )
                        )


                        if (
                            first_value
                            is not None
                            and first_value != 0
                        ):

                            normalized_market[
                                market_name
                            ] = (
                                close
                                / first_value
                                * 100
                            )


        if not normalized_market.empty:

            st.subheader(
                "市場比較（開始日=100）"
            )

            st.line_chart(
                normalized_market
            )


        st.success(
            "市場データ取得完了"
        )


        # ====================================================
        # ③ テクニカル分析
        # ====================================================

        st.header(
            "③ 📐 テクニカル分析"
        )


        technical_data = (
            add_all_indicators(
                stock_data.copy()
            )
        )


        latest_technical = (
            technical_data.iloc[-1]
        )


        technical_cols = st.columns(
            4
        )


        technical_cols[
            0
        ].metric(
            "SMA 5",
            safe_metric_number(
                latest_technical.get(
                    "SMA_5"
                ),
                0,
                " 円",
            ),
        )


        technical_cols[
            1
        ].metric(
            "SMA 25",
            safe_metric_number(
                latest_technical.get(
                    "SMA_25"
                ),
                0,
                " 円",
            ),
        )


        technical_cols[
            2
        ].metric(
            "SMA 75",
            safe_metric_number(
                latest_technical.get(
                    "SMA_75"
                ),
                0,
                " 円",
            ),
        )


        technical_cols[
            3
        ].metric(
            "RSI 14",
            safe_metric_number(
                latest_technical.get(
                    "RSI_14"
                ),
                2,
            ),
        )


        technical_chart_columns = [
            column

            for column in [
                "Close",
                "SMA_5",
                "SMA_25",
                "SMA_75",
            ]

            if column
            in technical_data.columns
        ]


        if technical_chart_columns:

            st.line_chart(
                technical_data[
                    technical_chart_columns
                ]
            )


        st.success(
            "テクニカル分析完了"
        )


        # ====================================================
        # ④ AI特徴量
        # ====================================================

        st.header(
            "④ 🧩 AI特徴量"
        )


        with st.spinner(
            "AI特徴量を作成しています..."
        ):

            ai_data = (
                build_ai_features(
                    technical_data,
                    market_data,
                )
            )


        if (
            ai_data is None
            or ai_data.empty
        ):

            st.error(
                "AI特徴量を作成できませんでした。"
            )

            st.stop()


        st.write(
            f"AI特徴量データ数: "
            f"{len(ai_data):,} 行"
        )


        market_feature_columns = [
            column

            for column
            in ai_data.columns

            if (
                "日経平均_" in column
                or "NASDAQ_" in column
                or "SOXX_" in column
                or "ドル円_" in column
            )
        ]


        st.write(
            "市場AI特徴量数:",
            len(
                market_feature_columns
            ),
        )


        if market_feature_columns:

            with st.expander(
                "市場AI特徴量を確認"
            ):

                st.dataframe(
                    ai_data[
                        market_feature_columns
                    ].tail(10),
                    use_container_width=True,
                )


        st.success(
            "AI特徴量作成完了"
        )


        # ====================================================
        # ⑤ AI学習
        # ====================================================

        st.header(
            "⑤ 🤖 AIモデル学習"
        )


        with st.spinner(
            "AIモデルを学習しています..."
        ):

            ai_model = (
                StockPredictionModel()
            )


            ai_metrics = (
                ai_model.train(
                    ai_data
                )
            )


        ai_metric_columns = st.columns(
            4
        )


        ai_metric_columns[
            0
        ].metric(
            "Accuracy",
            safe_metric_number(
                ai_metrics.get(
                    "accuracy"
                ),
                3,
            ),
        )


        ai_metric_columns[
            1
        ].metric(
            "Precision",
            safe_metric_number(
                ai_metrics.get(
                    "precision"
                ),
                3,
            ),
        )


        ai_metric_columns[
            2
        ].metric(
            "Recall",
            safe_metric_number(
                ai_metrics.get(
                    "recall"
                ),
                3,
            ),
        )


        ai_metric_columns[
            3
        ].metric(
            "F1",
            safe_metric_number(
                ai_metrics.get(
                    "f1"
                ),
                3,
            ),
        )


        st.success(
            "AIモデル学習完了"
        )


        # ====================================================
        # ⑥ 最新AI予測
        # ====================================================

        st.header(
            "⑥ 🔮 最新AI予測"
        )


        latest_probability = (
            ai_model.predict_probability(
                ai_data
            )
        )


        latest_probability = (
            safe_float(
                latest_probability
            )
        )


        if latest_probability is None:

            st.warning(
                "最新AI予測を取得できませんでした。"
            )

        else:

            prediction_label = (
                "上昇予測"
                if latest_probability
                >= 0.50
                else "下落予測"
            )


            prediction_cols = st.columns(
                2
            )


            prediction_cols[
                0
            ].metric(
                "翌営業日 上昇確率",
                f"{latest_probability * 100:.1f}%",
            )


            prediction_cols[
                1
            ].metric(
                "AI判断",
                prediction_label,
            )


        # ====================================================
        # ⑦ 特徴量重要度
        # ====================================================

        st.header(
            "⑦ 🧠 AI特徴量重要度"
        )


        try:

            importance = (
                ai_model
                .get_feature_importance()
            )


            if (
                importance is not None
                and not importance.empty
            ):

                st.dataframe(
                    importance.head(20),
                    use_container_width=True,
                )

            else:

                st.info(
                    "特徴量重要度データがありません。"
                )


        except Exception as error:

            st.info(
                f"特徴量重要度を表示できません: "
                f"{error}"
            )


        # ====================================================
        # ⑧ AIエントリー参考判定
        # ====================================================

        st.header(
            "⑧ 🎯 AIエントリー参考判定"
        )


        entry_strategy = (
            EntryStrategy()
        )


        entry_result = None


        if latest_probability is not None:

            try:

                entry_result = (
                    entry_strategy
                    .evaluate_latest(
                        ai_data,
                        latest_probability,
                    )
                )


                entry_cols = st.columns(
                    4
                )


                entry_cols[
                    0
                ].metric(
                    "判断",
                    entry_result.get(
                        "action",
                        "N/A",
                    ),
                )


                entry_cols[
                    1
                ].metric(
                    "スコア",
                    (
                        f"{entry_result.get('score', 0)}"
                        f" / "
                        f"{entry_result.get('max_score', 13)}"
                    ),
                )


                entry_cols[
                    2
                ].metric(
                    "AI確率",
                    (
                        f"{latest_probability * 100:.1f}%"
                    ),
                )


                entry_cols[
                    3
                ].metric(
                    "最低必要スコア",
                    entry_result.get(
                        "minimum_score",
                        6,
                    ),
                )


                reasons = (
                    entry_result.get(
                        "reasons",
                        [],
                    )
                )


                if reasons:

                    with st.expander(
                        "エントリー判定の詳細"
                    ):

                        for reason in reasons:

                            st.write(
                                "・",
                                reason,
                            )


            except Exception as error:

                st.warning(
                    f"エントリー判定を表示できません: "
                    f"{error}"
                )


        # ====================================================
        # ⑨ ウォークフォワード
        # ====================================================

        st.header(
            "⑨ 🔁 ウォークフォワード検証"
        )


        with st.spinner(
            "時系列AI検証を実行しています..."
        ):

            walk_engine = (
                WalkForwardBacktest(
                    initial_train_size=500,
                    test_size=20,
                    retrain_every=20,
                    threshold=0.50,
                )
            )


            walk_results, walk_metrics = (
                walk_engine.run(
                    ai_data
                )
            )


        walk_samples = (
            len(
                walk_results
            )

            if walk_results
            is not None

            else 0
        )


        walk_cols = st.columns(
            5
        )


        walk_cols[
            0
        ].metric(
            "検証件数",
            f"{walk_samples:,}",
        )


        walk_cols[
            1
        ].metric(
            "Accuracy",
            safe_metric_number(
                walk_metrics.get(
                    "accuracy"
                ),
                3,
            ),
        )


        walk_cols[
            2
        ].metric(
            "Precision",
            safe_metric_number(
                walk_metrics.get(
                    "precision"
                ),
                3,
            ),
        )


        walk_cols[
            3
        ].metric(
            "Recall",
            safe_metric_number(
                walk_metrics.get(
                    "recall"
                ),
                3,
            ),
        )


        walk_cols[
            4
        ].metric(
            "AUC",
            safe_metric_number(
                walk_metrics.get(
                    "auc"
                ),
                3,
            ),
        )


        if (
            walk_results is not None
            and not walk_results.empty
        ):

            with st.expander(
                "ウォークフォワード予測結果"
            ):

                st.dataframe(
                    walk_results.tail(100),
                    use_container_width=True,
                )


        st.success(
            "ウォークフォワード検証完了"
        )


        # ====================================================
        # ⑩ 従来型売買バックテスト
        # ====================================================

        st.header(
            "⑩ 📋 従来型売買バックテスト"
        )


        old_backtest_metrics = {}


        try:

            old_report = (
                BacktestReport(
                    initial_capital=
                        INITIAL_CAPITAL,

                    lot_size=
                        100,

                    entry_threshold=
                        0.60,

                    commission_rate=
                        0.001,

                    slippage_rate=
                        0.001,
                )
            )


            old_report.run(
                stock_data=
                    stock_data,

                prediction_data=
                    walk_results,
            )


            old_backtest_metrics = (
                old_report
                .calculate_metrics()
            )


            old_cols = st.columns(
                4
            )


            old_cols[
                0
            ].metric(
                "最終資産",
                (
                    f"{safe_float(old_backtest_metrics.get('final_capital'), INITIAL_CAPITAL):,.0f} 円"
                ),
            )


            old_cols[
                1
            ].metric(
                "取引回数",
                int(
                    safe_float(
                        old_backtest_metrics.get(
                            "trade_count"
                        ),
                        0,
                    )
                ),
            )


            old_cols[
                2
            ].metric(
                "勝率",
                format_percent(
                    old_backtest_metrics.get(
                        "win_rate"
                    )
                ),
            )


            old_cols[
                3
            ].metric(
                "最大DD",
                format_percent(
                    old_backtest_metrics.get(
                        "max_drawdown"
                    )
                ),
            )


        except Exception as error:

            st.warning(
                f"従来型バックテストを表示できません: "
                f"{error}"
            )


        # ====================================================
        # ⑪ 最新EntryStrategy
        # ====================================================

        st.header(
            "⑪ 🚦 最新エントリー判断"
        )


        if entry_result is not None:

            st.write(
                entry_result.get(
                    "summary",
                    "",
                )
            )

        else:

            st.info(
                "最新エントリー判断はありません。"
            )


        # ====================================================
        # ⑫ リスク・資金管理
        # ====================================================

        st.header(
            "⑫ 🛡️ リスク・資金管理"
        )


        if latest_close is not None:

            risk_manager = (
                RiskManager(
                    lot_size=
                        100,

                    risk_per_trade=
                        0.01,

                    max_position_rate=
                        0.50,

                    stop_loss_rate=
                        0.05,

                    take_profit_rate=
                        0.10,

                    commission_rate=
                        0.001,

                    slippage_rate=
                        0.001,
                )
            )


            risk_result = (
                risk_manager.evaluate_trade(
                    capital=
                        INITIAL_CAPITAL,

                    market_price=
                        latest_close,
                )
            )


            risk_cols = st.columns(
                4
            )


            risk_cols[
                0
            ].metric(
                "購入可能",
                (
                    "YES"
                    if risk_result.get(
                        "can_trade",
                        False,
                    )
                    else "NO"
                ),
            )


            risk_cols[
                1
            ].metric(
                "推奨株数",
                (
                    f"{int(risk_result.get('shares', 0)):,} 株"
                ),
            )


            risk_cols[
                2
            ].metric(
                "損切り価格",
                safe_metric_number(
                    risk_result.get(
                        "stop_price"
                    ),
                    0,
                    " 円",
                ),
            )


            risk_cols[
                3
            ].metric(
                "利益確定価格",
                safe_metric_number(
                    risk_result.get(
                        "take_profit_price"
                    ),
                    0,
                    " 円",
                ),
            )


            st.caption(
                risk_result.get(
                    "reason",
                    "",
                )
            )


        # ====================================================
        # ⑬ 本格戦略バックテスト
        # ====================================================

        st.header(
            "⑬ 🚀 本格戦略バックテスト"
        )


        st.write(
            "EntryStrategy・ExitStrategy・RiskManagerを"
            "統合して過去データを時系列で検証します。"
        )


        st.write(
            "当日の終値でBUY/SELLを判定し、"
            "実際の売買は翌営業日の始値で行います。"
        )


        # ====================================================
        # ⑬ バックテスト設定
        # ====================================================

        st.subheader(
            "⚙️ バックテスト設定"
        )


        setting_col1, setting_col2 = (
            st.columns(
                2
            )
        )


        # ----------------------------------------------------
        # 初期資金
        # ----------------------------------------------------

        capital_options = {
            "100万円":
                1_000_000,

            "300万円":
                3_000_000,

            "500万円":
                5_000_000,

            "1,000万円":
                10_000_000,
        }


        selected_capital_label = (
            setting_col1.selectbox(
                "バックテスト資金",
                options=list(
                    capital_options.keys()
                ),
                index=0,
                help=(
                    "⑬の本格戦略バックテストだけに"
                    "使用する仮想運用資金です。"
                ),
            )
        )


        backtest_capital = (
            capital_options[
                selected_capital_label
            ]
        )


        # ----------------------------------------------------
        # 最大投資比率
        # ----------------------------------------------------

        position_rate_options = {
            "25%":
                0.25,

            "50%":
                0.50,

            "75%":
                0.75,

            "100%":
                1.00,
        }


        selected_position_label = (
            setting_col2.selectbox(
                "最大投資比率",
                options=list(
                    position_rate_options.keys()
                ),
                index=1,
                help=(
                    "総資産のうち1回のポジションに"
                    "使用できる最大割合です。"
                ),
            )
        )


        max_position_rate = (
            position_rate_options[
                selected_position_label
            ]
        )


        # ----------------------------------------------------
        # 設定内容表示
        # ----------------------------------------------------

        maximum_position_value = (
            backtest_capital
            * max_position_rate
        )


        setting_info_cols = (
            st.columns(
                3
            )
        )


        setting_info_cols[
            0
        ].metric(
            "設定資金",
            (
                f"{backtest_capital:,.0f} 円"
            ),
        )


        setting_info_cols[
            1
        ].metric(
            "最大投資比率",
            selected_position_label,
        )


        setting_info_cols[
            2
        ].metric(
            "1回の最大投資額",
            (
                f"{maximum_position_value:,.0f} 円"
            ),
        )


        st.caption(
            "売買単位は100株、"
            "1回の許容リスクは資金の1%、"
            "損切り5%、利益確定10%の"
            "初期設定で検証します。"
        )


        # ====================================================
        # ⑬ 実行
        # ====================================================

        with st.spinner(
            "本格戦略バックテストを実行しています..."
        ):

            strategy_engine = (
                TradingBacktestEngine(
                    initial_capital=
                        backtest_capital,

                    lot_size=
                        100,

                    entry_minimum_score=
                        6,

                    entry_minimum_probability=
                        0.55,

                    entry_strong_probability=
                        0.60,

                    stop_loss_rate=
                        0.05,

                    take_profit_rate=
                        0.10,

                    ai_exit_probability=
                        0.45,

                    max_holding_days=
                        10,

                    minimum_exit_score=
                        3,

                    risk_per_trade=
                        0.01,

                    max_position_rate=
                        max_position_rate,

                    commission_rate=
                        0.001,

                    slippage_rate=
                        0.001,
                )
            )


            (
                strategy_trades,
                strategy_equity,
                strategy_metrics,
            ) = (
                strategy_engine.run(
                    stock_data=
                        stock_data,

                    ai_data=
                        ai_data,

                    walk_results=
                        walk_results,
                )
            )


        # ====================================================
        # 総合成績
        # ====================================================

        st.subheader(
            "📊 総合成績"
        )


        initial_value = safe_float(
            strategy_metrics.get(
                "initial_capital"
            ),
            backtest_capital,
        )


        final_value = safe_float(
            strategy_metrics.get(
                "final_capital"
            ),
            backtest_capital,
        )


        total_profit = safe_float(
            strategy_metrics.get(
                "total_profit"
            ),
            0.0,
        )


        total_return = safe_float(
            strategy_metrics.get(
                "total_return"
            ),
            0.0,
        )


        result_cols = st.columns(
            4
        )


        result_cols[
            0
        ].metric(
            "初期資金",
            (
                f"{initial_value:,.0f} 円"
            ),
        )


        result_cols[
            1
        ].metric(
            "最終資産",
            (
                f"{final_value:,.0f} 円"
            ),
            delta=(
                f"{total_profit:+,.0f} 円"
            ),
        )


        result_cols[
            2
        ].metric(
            "総利益",
            (
                f"{total_profit:+,.0f} 円"
            ),
        )


        result_cols[
            3
        ].metric(
            "総合リターン",
            (
                f"{total_return * 100:+.2f}%"
            ),
        )


        # ====================================================
        # 取引成績
        # ====================================================

        st.subheader(
            "🎯 取引成績"
        )


        trade_count = int(
            safe_float(
                strategy_metrics.get(
                    "trade_count",
                    strategy_metrics.get(
                        "trades",
                        0,
                    ),
                ),
                0,
            )
        )


        wins = int(
            safe_float(
                strategy_metrics.get(
                    "wins"
                ),
                0,
            )
        )


        losses = int(
            safe_float(
                strategy_metrics.get(
                    "losses"
                ),
                0,
            )
        )


        win_rate = safe_float(
            strategy_metrics.get(
                "win_rate"
            )
        )


        trade_cols = st.columns(
            4
        )


        trade_cols[
            0
        ].metric(
            "取引回数",
            trade_count,
        )


        trade_cols[
            1
        ].metric(
            "勝ち",
            wins,
        )


        trade_cols[
            2
        ].metric(
            "負け",
            losses,
        )


        trade_cols[
            3
        ].metric(
            "勝率",
            (
                f"{win_rate * 100:.1f}%"

                if win_rate
                is not None

                else "N/A"
            ),
        )


        # ====================================================
        # リスク・効率
        # ====================================================

        st.subheader(
            "🛡️ リスク・効率"
        )


        profit_factor = safe_float(
            strategy_metrics.get(
                "profit_factor"
            )
        )


        max_drawdown = safe_float(
            strategy_metrics.get(
                "max_drawdown"
            ),
            0.0,
        )


        average_holding = safe_float(
            strategy_metrics.get(
                "average_holding_days"
            )
        )


        skipped_count = int(
            safe_float(
                strategy_metrics.get(
                    "skipped_entries"
                ),
                0,
            )
        )


        total_commission = safe_float(
            strategy_metrics.get(
                "total_commission"
            ),
            0.0,
        )


        risk_cols = st.columns(
            5
        )


        risk_cols[
            0
        ].metric(
            "プロフィットファクター",
            (
                f"{profit_factor:.2f}"

                if profit_factor
                is not None
                and np.isfinite(
                    profit_factor
                )

                else (
                    "∞"
                    if profit_factor
                    is not None
                    and np.isinf(
                        profit_factor
                    )
                    else "N/A"
                )
            ),
        )


        risk_cols[
            1
        ].metric(
            "最大ドローダウン",
            (
                f"{max_drawdown * 100:.2f}%"
            ),
        )


        risk_cols[
            2
        ].metric(
            "平均保有日数",
            (
                f"{average_holding:.1f} 日"

                if average_holding
                is not None

                else "N/A"
            ),
        )


        risk_cols[
            3
        ].metric(
            "資金管理で見送り",
            (
                f"{skipped_count:,} 回"
            ),
        )


        risk_cols[
            4
        ].metric(
            "累計売買手数料",
            (
                f"{total_commission:,.0f} 円"
            ),
        )


        # ====================================================
        # 資産曲線
        # ====================================================

        if (
            strategy_equity is not None
            and not strategy_equity.empty
            and "Total_Equity"
            in strategy_equity.columns
        ):

            st.subheader(
                "📈 資産推移"
            )


            st.line_chart(
                strategy_equity[
                    ["Total_Equity"]
                ]
            )


        # ====================================================
        # 売買履歴
        # ====================================================

        if (
            strategy_trades is not None
            and not strategy_trades.empty
        ):

            st.subheader(
                "🧾 売買履歴"
            )


            st.dataframe(
                strategy_trades,
                use_container_width=True,
            )


        else:

            st.info(
                "この設定では実際に成立した売買はありません。"
            )


        # ====================================================
        # 資金管理で見送ったBUY
        # ====================================================

        try:

            skipped_entries = (
                strategy_engine
                .get_skipped_entries()
            )


            if (
                skipped_entries is not None
                and not skipped_entries.empty
            ):

                with st.expander(
                    "🔍 資金管理で見送ったBUY候補"
                ):

                    st.dataframe(
                        skipped_entries,
                        use_container_width=True,
                    )


        except Exception:

            pass


        # ====================================================
        # 注文ログ
        # ====================================================

        try:

            order_log = (
                strategy_engine
                .get_order_log()
            )


            if (
                order_log is not None
                and not order_log.empty
            ):

                with st.expander(
                    "📋 BUY / SELL シグナル・注文ログ"
                ):

                    st.dataframe(
                        order_log,
                        use_container_width=True,
                    )


        except Exception:

            pass


        # ====================================================
        # 完了
        # ====================================================

        st.success(
            "🎉 ⑬ 本格戦略バックテストまで"
            "正常に完了しました。"
        )


        st.info(
            "この結果は過去データを使った"
            "バックテストであり、"
            "将来の利益を保証するものではありません。"
        )


    # ========================================================
    # 全体エラー
    # ========================================================

    except Exception as error:

        st.error(
            "分析処理中にエラーが発生しました。"
        )


        st.exception(
            error
        )


# ============================================================
# 未実行時
# ============================================================

else:

    st.info(
        "「🚀 株価・市場・AI・戦略分析を実行」"
        "を押すと分析を開始します。"
    )
