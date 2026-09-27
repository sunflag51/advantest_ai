# ============================================================
# アドバンテスト AI売買システム
# main.py
#
# 1. 株価取得
# 2. 市場データ取得
# 3. テクニカル分析
# 4. AI特徴量
# 5. AI学習
# 6. 最新AI予測
# 7. 特徴量重要度
# 8. 固定80/20検証
# 9. ウォークフォワード検証
# 10. 正式な売買バックテスト
# 11. AIエントリー判断
#
# ※実際の売買注文は行いません
# ============================================================


import streamlit as st
import pandas as pd


# ============================================================
# データ
# ============================================================

from data.stock_data import get_stock_data

from data.market_data import (
    get_all_market_data,
    get_latest_market_values
)


# ============================================================
# テクニカル分析
# ============================================================

from indicators.technical import (
    add_all_indicators,
    get_latest_indicators
)


# ============================================================
# AI特徴量
# ============================================================

from ai.features import (
    build_ai_features,
    get_ai_feature_columns,
    get_feature_summary,
    get_available_market_features
)


# ============================================================
# AIモデル
# ============================================================

from ai.model import StockPredictionModel


# ============================================================
# バックテスト
# ============================================================

from backtest.engine import WalkForwardBacktest
from backtest.report import BacktestReport


# ============================================================
# 売買戦略
# ============================================================

from strategy.entry import EntryStrategy


# ============================================================
# 基本設定
# ============================================================

STOCK_CODE = "6857.T"
STOCK_NAME = "アドバンテスト"


# ============================================================
# Streamlit設定
# ============================================================

st.set_page_config(
    page_title="アドバンテスト AI売買システム",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# タイトル
# ============================================================

st.title(
    "📈 アドバンテスト AI売買システム"
)

st.write(
    "株価・市場環境・テクニカル指標・AIを統合して、"
    "予測・バックテスト・エントリー判断を行います。"
)

st.caption(
    "現在は研究・検証段階です。"
    "実際の売買注文は行いません。"
)

st.divider()


# ============================================================
# システム状態
# ============================================================

st.subheader(
    "システム状態"
)


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.success("✅ 株価")


with col2:
    st.success("✅ 市場")


with col3:
    st.success("🤖 AI")


with col4:
    st.success("🧪 戦略検証")


# ============================================================
# 実行ボタン
# ============================================================

st.divider()

st.subheader(
    "🚀 分析開始"
)


run_analysis = st.button(
    "株価・市場・AI・戦略分析を実行",
    type="primary",
    use_container_width=True
)


# ============================================================
# 分析開始
# ============================================================

if run_analysis:

    # ========================================================
    # ① 株価
    # ========================================================

    st.header(
        "① アドバンテスト株価"
    )


    try:

        with st.spinner(
            "株価を取得しています..."
        ):

            stock_data = get_stock_data(
                stock_code=STOCK_CODE,
                period="5y",
                interval="1d"
            )


        if stock_data is None or stock_data.empty:

            st.error(
                "株価データを取得できませんでした。"
            )

            st.stop()


        st.success(
            "✅ 株価取得成功"
        )


        latest_stock = stock_data.iloc[-1]


        close_price = float(
            latest_stock["Close"]
        )

        open_price = float(
            latest_stock["Open"]
        )

        high_price = float(
            latest_stock["High"]
        )

        low_price = float(
            latest_stock["Low"]
        )

        volume = int(
            latest_stock["Volume"]
        )


        if len(stock_data) >= 2:

            previous_close = float(
                stock_data.iloc[-2]["Close"]
            )

            change = (
                close_price
                - previous_close
            )

            change_rate = (
                change
                / previous_close
                * 100
            )

        else:

            change = None
            change_rate = None


        col1, col2, col3 = st.columns(3)


        with col1:

            if change is not None:

                st.metric(
                    "終値",
                    f"{close_price:,.0f} 円",
                    delta=(
                        f"{change:+,.0f}円 "
                        f"({change_rate:+.2f}%)"
                    )
                )

            else:

                st.metric(
                    "終値",
                    f"{close_price:,.0f} 円"
                )


        with col2:

            st.metric(
                "始値",
                f"{open_price:,.0f} 円"
            )


        with col3:

            st.metric(
                "出来高",
                f"{volume:,}"
            )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "高値",
                f"{high_price:,.0f} 円"
            )


        with col2:

            st.metric(
                "安値",
                f"{low_price:,.0f} 円"
            )


        st.line_chart(
            stock_data[
                ["Close"]
            ].tail(250)
        )


    except Exception as e:

        st.error(
            "❌ 株価取得エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ② 市場
    # ========================================================

    st.divider()

    st.header(
        "② 市場環境"
    )


    try:

        with st.spinner(
            "市場データを取得しています..."
        ):

            market_data = get_all_market_data(
                period="5y",
                interval="1d"
            )

            latest_market = get_latest_market_values(
                period="5d"
            )


        st.success(
            "✅ 市場データ取得成功"
        )


        market_names = [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円"
        ]


        cols = st.columns(4)


        for i, market_name in enumerate(
            market_names
        ):

            market_value = latest_market.get(
                market_name
            )


            with cols[i]:

                if (
                    market_value
                    and market_value.get(
                        "close"
                    ) is not None
                ):

                    st.metric(
                        market_name,
                        f"{market_value['close']:,.2f}"
                    )

                else:

                    st.warning(
                        f"{market_name} データなし"
                    )


        # ----------------------------------------------------
        # 市場比較チャート
        # ----------------------------------------------------

        market_chart = pd.DataFrame()


        for market_name in market_names:

            if market_name not in market_data:

                continue


            market_df = market_data[
                market_name
            ]


            if (
                market_df is None
                or market_df.empty
                or "Close" not in market_df.columns
            ):

                continue


            series = pd.to_numeric(
                market_df["Close"],
                errors="coerce"
            ).dropna()


            if series.empty:

                continue


            normalized = (
                series
                / series.iloc[0]
                * 100
            )

            normalized.name = market_name


            if market_chart.empty:

                market_chart = (
                    normalized.to_frame()
                )

            else:

                market_chart = (
                    market_chart.join(
                        normalized,
                        how="outer"
                    )
                )


        if not market_chart.empty:

            market_chart = (
                market_chart
                .sort_index()
                .ffill()
            )


            st.subheader(
                "🌏 市場比較"
            )

            st.caption(
                "表示開始時点を100として比較しています。"
            )

            st.line_chart(
                market_chart.tail(250)
            )


    except Exception as e:

        st.error(
            "❌ 市場データ取得エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ③ テクニカル分析
    # ========================================================

    st.divider()

    st.header(
        "③ テクニカル分析"
    )


    try:

        technical_data = add_all_indicators(
            stock_data
        )


        latest_indicators = get_latest_indicators(
            technical_data
        )


        st.success(
            "✅ テクニカル分析完了"
        )


        col1, col2, col3 = st.columns(3)


        sma5 = latest_indicators.get(
            "SMA_5"
        )

        sma25 = latest_indicators.get(
            "SMA_25"
        )

        sma75 = latest_indicators.get(
            "SMA_75"
        )


        with col1:

            if pd.notna(sma5):

                st.metric(
                    "5日移動平均",
                    f"{sma5:,.0f} 円"
                )


        with col2:

            if pd.notna(sma25):

                st.metric(
                    "25日移動平均",
                    f"{sma25:,.0f} 円"
                )


        with col3:

            if pd.notna(sma75):

                st.metric(
                    "75日移動平均",
                    f"{sma75:,.0f} 円"
                )


        ma_columns = [
            column
            for column in [
                "Close",
                "SMA_5",
                "SMA_25",
                "SMA_75"
            ]
            if column in technical_data.columns
        ]


        st.line_chart(
            technical_data[
                ma_columns
            ].tail(250)
        )


        # ----------------------------------------------------
        # RSI / MACD
        # ----------------------------------------------------

        rsi = latest_indicators.get(
            "RSI_14"
        )

        macd = latest_indicators.get(
            "MACD"
        )

        macd_signal = latest_indicators.get(
            "MACD_Signal"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if pd.notna(rsi):

                st.metric(
                    "RSI",
                    f"{rsi:.2f}"
                )


        with col2:

            if pd.notna(macd):

                st.metric(
                    "MACD",
                    f"{macd:.2f}"
                )


        with col3:

            if pd.notna(macd_signal):

                st.metric(
                    "MACD Signal",
                    f"{macd_signal:.2f}"
                )


        # ----------------------------------------------------
        # ATR / 出来高 / ボラティリティ
        # ----------------------------------------------------

        atr = latest_indicators.get(
            "ATR_14"
        )

        volume_ratio = latest_indicators.get(
            "Volume_Ratio"
        )

        volatility = latest_indicators.get(
            "Volatility_20D"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if pd.notna(atr):

                st.metric(
                    "ATR",
                    f"{atr:,.2f}"
                )


        with col2:

            if pd.notna(volume_ratio):

                st.metric(
                    "出来高比率",
                    f"{volume_ratio:.2f}"
                )


        with col3:

            if pd.notna(volatility):

                st.metric(
                    "20日ボラティリティ",
                    f"{volatility * 100:.2f}%"
                )


    except Exception as e:

        st.error(
            "❌ テクニカル分析エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ④ AI特徴量
    # ========================================================

    st.divider()

    st.header(
        "④ 🧠 AI特徴量"
    )


    try:

        ai_data = build_ai_features(
            technical_data=technical_data,
            market_data=market_data
        )


        ai_feature_columns = get_ai_feature_columns(
            ai_data
        )


        feature_summary = get_feature_summary(
            ai_data
        )


        market_feature_status = (
            get_available_market_features(
                ai_data
            )
        )


        st.success(
            "✅ AI特徴量作成完了"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "特徴量数",
                len(ai_feature_columns)
            )


        with col2:

            st.metric(
                "データ行数",
                feature_summary.get(
                    "rows",
                    0
                )
            )


        with col3:

            st.metric(
                "欠損値数",
                feature_summary.get(
                    "missing_values",
                    0
                )
            )


        market_rows = []


        for market_name in market_names:

            status = (
                market_feature_status.get(
                    market_name,
                    {}
                )
            )


            market_rows.append(
                {
                    "市場":
                        market_name,

                    "状態":
                        (
                            "✅ 使用"
                            if status.get(
                                "available",
                                False
                            )
                            else "⚠️ 未使用"
                        ),

                    "特徴量数":
                        status.get(
                            "feature_count",
                            0
                        )
                }
            )


        st.dataframe(
            pd.DataFrame(
                market_rows
            ),
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ AI特徴量作成エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑤ AI学習
    # ========================================================

    st.divider()

    st.header(
        "⑤ 🤖 市場データ対応AI"
    )


    try:

        ai_model = StockPredictionModel()


        with st.spinner(
            "AIを学習しています..."
        ):

            metrics = ai_model.train(
                data=ai_data,
                feature_columns=ai_feature_columns,
                train_ratio=0.8
            )


        st.success(
            "✅ AI学習完了"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Accuracy",
                f"{metrics.get('accuracy', 0) * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Precision",
                f"{metrics.get('precision', 0) * 100:.2f}%"
            )


        with col3:

            st.metric(
                "Recall",
                f"{metrics.get('recall', 0) * 100:.2f}%"
            )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "F1",
                f"{metrics.get('f1', 0) * 100:.2f}%"
            )


        with col2:

            auc = metrics.get(
                "auc"
            )


            st.metric(
                "AUC",
                (
                    f"{auc:.3f}"
                    if auc is not None
                    else "計算不可"
                )
            )


    except Exception as e:

        st.error(
            "❌ AI学習エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑥ 最新AI予測
    # ========================================================

    st.divider()

    st.header(
        "⑥ 🔮 最新AI予測"
    )


    try:

        prediction = ai_model.predict(
            ai_data,
            threshold=0.50
        )


        probability_up = float(
            prediction[
                "probability_up"
            ]
        )


        probability_down = float(
            prediction[
                "probability_down"
            ]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📈 上昇確率",
                f"{probability_up * 100:.2f}%"
            )


        with col2:

            st.metric(
                "📉 下落確率",
                f"{probability_down * 100:.2f}%"
            )


        st.progress(
            min(
                max(
                    probability_up,
                    0.0
                ),
                1.0
            )
        )


        if prediction[
            "prediction_text"
        ] == "上昇":

            st.success(
                "AI方向予測：上昇側"
            )

        else:

            st.warning(
                "AI方向予測：下落側"
            )


        st.caption(
            "このAI確率は⑪のエントリー判断にも使用します。"
        )


    except Exception as e:

        st.error(
            "❌ 最新AI予測エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑦ 特徴量重要度
    # ========================================================

    st.divider()

    st.header(
        "⑦ 🔍 AI特徴量重要度"
    )


    try:

        importance = (
            ai_model
            .get_feature_importance()
        )


        importance_display = (
            importance.copy()
        )


        importance_display[
            "importance"
        ] *= 100


        importance_display.rename(
            columns={
                "feature":
                    "特徴量",

                "importance":
                    "重要度（%）"
            },
            inplace=True
        )


        top_importance = (
            importance_display
            .head(15)
        )


        st.dataframe(
            top_importance,
            use_container_width=True,
            hide_index=True
        )


        st.bar_chart(
            top_importance
            .set_index(
                "特徴量"
            )[
                ["重要度（%）"]
            ]
        )


    except Exception as e:

        st.warning(
            "特徴量重要度を表示できませんでした。"
        )

        st.exception(e)


    # ========================================================
    # ⑧ 固定80/20テスト
    # ========================================================

    st.divider()

    st.header(
        "⑧ 📋 固定80/20テスト"
    )


    try:

        test_results = (
            ai_model
            .get_test_results()
        )


        if (
            test_results is not None
            and not test_results.empty
        ):

            display_test = (
                test_results.copy()
            )


            display_test[
                "上昇確率（%）"
            ] = (
                display_test[
                    "Probability_Up"
                ]
                * 100
            )


            display_test[
                "実際"
            ] = (
                display_test[
                    "Actual"
                ]
                .map(
                    {
                        1: "上昇",
                        0: "下落"
                    }
                )
            )


            display_test[
                "AI予測"
            ] = (
                display_test[
                    "Prediction"
                ]
                .map(
                    {
                        1: "上昇",
                        0: "下落"
                    }
                )
            )


            st.dataframe(
                display_test[
                    [
                        "実際",
                        "AI予測",
                        "上昇確率（%）"
                    ]
                ].tail(50),
                use_container_width=True
            )


    except Exception as e:

        st.warning(
            "固定テスト結果を表示できませんでした。"
        )

        st.exception(e)


    # ========================================================
    # ⑨ ウォークフォワード検証
    # ========================================================

    st.divider()

    st.header(
        "⑨ 🔄 ウォークフォワード検証"
    )


    try:

        with st.spinner(
            "ウォークフォワード検証中..."
        ):

            walk_forward = (
                WalkForwardBacktest(
                    initial_train_size=500,
                    test_size=20,
                    retrain_every=20,
                    threshold=0.50
                )
            )


            (
                walk_results,
                walk_metrics
            ) = walk_forward.run(
                data=ai_data,
                feature_columns=ai_feature_columns
            )


        st.success(
            "✅ ウォークフォワード検証完了"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Accuracy",
                f"{walk_metrics.get('accuracy', 0) * 100:.2f}%"
            )


        with col2:

            walk_auc = (
                walk_metrics.get(
                    "auc"
                )
            )


            st.metric(
                "AUC",
                (
                    f"{walk_auc:.3f}"
                    if walk_auc is not None
                    else "計算不可"
                )
            )


        with col3:

            st.metric(
                "検証営業日",
                f"{walk_metrics.get('samples', 0):,}"
            )


        high_accuracy = (
            walk_metrics.get(
                "high_confidence_accuracy"
            )
        )

        high_return = (
            walk_metrics.get(
                "high_confidence_return"
            )
        )


        st.subheader(
            "🎯 上昇確率60%以上"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "予測回数",
                walk_metrics.get(
                    "high_confidence_samples",
                    0
                )
            )


        with col2:

            st.metric(
                "正解率",
                (
                    f"{high_accuracy * 100:.2f}%"
                    if high_accuracy is not None
                    else "データなし"
                )
            )


        with col3:

            st.metric(
                "平均翌日リターン",
                (
                    f"{high_return * 100:+.3f}%"
                    if high_return is not None
                    else "データなし"
                )
            )


    except Exception as e:

        st.error(
            "❌ ウォークフォワード検証エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑩ 正式な売買バックテスト
    # ========================================================

    st.divider()

    st.header(
        "⑩ 💰 正式な売買バックテスト"
    )


    try:

        with st.spinner(
            "売買バックテストを実行しています..."
        ):

            backtest_report = (
                BacktestReport(
                    initial_capital=1_000_000,
                    lot_size=100,
                    entry_threshold=0.60,
                    commission_rate=0.001,
                    slippage_rate=0.001
                )
            )


            (
                trades,
                equity_curve,
                backtest_metrics
            ) = backtest_report.run(
                stock_data=stock_data,
                walk_results=walk_results
            )


        st.success(
            "✅ 売買バックテスト完了"
        )


        # ----------------------------------------------------
        # 資産
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "初期資金",
                f"{backtest_metrics.get('initial_capital', 0):,.0f} 円"
            )


        with col2:

            final_capital = (
                backtest_metrics.get(
                    "final_capital",
                    0
                )
            )

            total_profit = (
                backtest_metrics.get(
                    "total_profit",
                    0
                )
            )


            st.metric(
                "最終資産",
                f"{final_capital:,.0f} 円",
                delta=(
                    f"{total_profit:+,.0f} 円"
                )
            )


        # ----------------------------------------------------
        # 成績
        # ----------------------------------------------------

        total_return = (
            backtest_metrics.get(
                "total_return",
                0
            )
        )

        win_rate = (
            backtest_metrics.get(
                "win_rate"
            )
        )

        profit_factor = (
            backtest_metrics.get(
                "profit_factor"
            )
        )

        max_drawdown = (
            backtest_metrics.get(
                "max_drawdown",
                0
            )
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "総リターン",
                f"{total_return * 100:+.2f}%"
            )


        with col2:

            st.metric(
                "勝率",
                (
                    f"{win_rate * 100:.2f}%"
                    if win_rate is not None
                    else "取引なし"
                )
            )


        with col3:

            st.metric(
                "Profit Factor",
                (
                    f"{profit_factor:.3f}"
                    if profit_factor is not None
                    else "計算不可"
                )
            )


        with col4:

            st.metric(
                "最大ドローダウン",
                f"{max_drawdown * 100:.2f}%"
            )


        st.metric(
            "取引回数",
            backtest_metrics.get(
                "trades",
                0
            )
        )


        # ----------------------------------------------------
        # Buy & Hold比較
        # ----------------------------------------------------

        buy_hold_return = (
            backtest_metrics.get(
                "buy_hold_return"
            )
        )

        excess_return = (
            backtest_metrics.get(
                "excess_return"
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Buy & Hold",
                (
                    f"{buy_hold_return * 100:+.2f}%"
                    if buy_hold_return is not None
                    else "計算不可"
                )
            )


        with col2:

            st.metric(
                "AI戦略との差",
                (
                    f"{excess_return * 100:+.2f}%"
                    if excess_return is not None
                    else "計算不可"
                )
            )


        # ----------------------------------------------------
        # 資産曲線
        # ----------------------------------------------------

        comparison_curve = (
            backtest_report
            .get_comparison_curve()
        )


        if (
            comparison_curve is not None
            and not comparison_curve.empty
        ):

            comparison_display = (
                comparison_curve.copy()
            )


            comparison_display.rename(
                columns={
                    "AI_Strategy":
                        "AI戦略",

                    "BuyHold_Equity":
                        "Buy & Hold"
                },
                inplace=True
            )


            st.line_chart(
                comparison_display
            )


        # ----------------------------------------------------
        # 取引履歴
        # ----------------------------------------------------

        with st.expander(
            "バックテスト取引履歴を見る"
        ):

            if (
                trades is None
                or trades.empty
            ):

                st.warning(
                    "取引がありません。"
                )

            else:

                trade_display = (
                    trades.copy()
                )


                trade_display[
                    "上昇確率（%）"
                ] = (
                    trade_display[
                        "Probability_Up"
                    ]
                    * 100
                )


                trade_display[
                    "損益率（%）"
                ] = (
                    trade_display[
                        "Profit_Rate"
                    ]
                    * 100
                )


                st.dataframe(
                    trade_display[
                        [
                            "Signal_Date",
                            "Trade_Date",
                            "上昇確率（%）",
                            "Buy_Price",
                            "Sell_Price",
                            "Shares",
                            "Profit",
                            "損益率（%）",
                            "Capital_After"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )


    except Exception as e:

        st.error(
            "❌ 売買バックテストエラー"
        )

        st.exception(e)


    # ========================================================
    # ⑪ AIエントリー判断
    # ========================================================

    st.divider()

    st.header(
        "⑪ 🎯 AIエントリー判断"
    )


    st.write(
        "最新のAI上昇確率・トレンド・RSI・MACD・"
        "出来高・市場環境を組み合わせて、"
        "BUY / WAITを判定します。"
    )


    try:

        # ----------------------------------------------------
        # エントリー戦略
        # ----------------------------------------------------

        entry_strategy = EntryStrategy(
            minimum_score=6,
            minimum_probability=0.55,
            strong_probability=0.60,
            rsi_min=40.0,
            rsi_max=70.0,
            volume_ratio_min=1.0
        )


        # ----------------------------------------------------
        # 最新日の判定
        # ----------------------------------------------------

        entry_result = (
            entry_strategy.evaluate_latest(
                data=ai_data,
                probability_up=probability_up
            )
        )


        entry_action = (
            entry_result[
                "action"
            ]
        )

        entry_score = (
            entry_result[
                "score"
            ]
        )

        max_score = (
            entry_result[
                "max_score"
            ]
        )

        score_rate = (
            entry_result[
                "score_rate"
            ]
        )

        minimum_score = (
            entry_result[
                "minimum_score"
            ]
        )

        entry_probability = (
            entry_result[
                "probability_up"
            ]
        )

        minimum_probability = (
            entry_result[
                "minimum_probability"
            ]
        )

        entry_summary = (
            entry_result[
                "summary"
            ]
        )

        entry_reasons = (
            entry_result[
                "reasons"
            ]
        )

        entry_details = (
            entry_result[
                "details"
            ]
        )

        entry_date = (
            entry_result.get(
                "date"
            )
        )


        # ----------------------------------------------------
        # 判定日
        # ----------------------------------------------------

        if entry_date is not None:

            try:

                st.caption(
                    "判定データ日："
                    f"{pd.Timestamp(entry_date).strftime('%Y-%m-%d')}"
                )

            except Exception:

                pass


        # ====================================================
        # BUY / WAIT 大表示
        # ====================================================

        if entry_action == "BUY":

            st.success(
                "🟢 エントリー判断：BUY"
            )

        else:

            st.warning(
                "🟡 エントリー判断：WAIT"
            )


        st.write(
            entry_summary
        )


        # ====================================================
        # メイン指標
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "AI上昇確率",
                f"{entry_probability * 100:.2f}%",
                delta=(
                    "基準 "
                    f"{minimum_probability * 100:.0f}%以上"
                )
            )


        with col2:

            st.metric(
                "総合スコア",
                f"{entry_score} / {max_score}",
                delta=(
                    f"BUY基準 {minimum_score}点以上"
                )
            )


        with col3:

            st.metric(
                "スコア率",
                f"{score_rate * 100:.1f}%"
            )


        # ----------------------------------------------------
        # スコア表示
        # ----------------------------------------------------

        st.write(
            "総合スコア"
        )


        st.progress(
            min(
                max(
                    score_rate,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # スコア内訳
        # ====================================================

        st.subheader(
            "📊 エントリースコア内訳"
        )


        score_breakdown = pd.DataFrame(
            [
                {
                    "判定項目":
                        "AI上昇確率",

                    "獲得点":
                        entry_details.get(
                            "ai_score",
                            0
                        ),

                    "最大点":
                        3
                },

                {
                    "判定項目":
                        "トレンド",

                    "獲得点":
                        entry_details.get(
                            "trend_score",
                            0
                        ),

                    "最大点":
                        3
                },

                {
                    "判定項目":
                        "RSI",

                    "獲得点":
                        entry_details.get(
                            "rsi_score",
                            0
                        ),

                    "最大点":
                        1
                },

                {
                    "判定項目":
                        "MACD",

                    "獲得点":
                        entry_details.get(
                            "macd_score",
                            0
                        ),

                    "最大点":
                        1
                },

                {
                    "判定項目":
                        "出来高",

                    "獲得点":
                        entry_details.get(
                            "volume_score",
                            0
                        ),

                    "最大点":
                        1
                },

                {
                    "判定項目":
                        "市場環境",

                    "獲得点":
                        entry_details.get(
                            "market_score",
                            0
                        ),

                    "最大点":
                        4
                }
            ]
        )


        score_breakdown[
            "達成率（%）"
        ] = (
            score_breakdown[
                "獲得点"
            ]
            / score_breakdown[
                "最大点"
            ]
            * 100
        )


        st.dataframe(
            score_breakdown,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 判定理由
        # ====================================================

        st.subheader(
            "🔍 判定理由"
        )


        for reason in entry_reasons:

            st.write(
                f"・{reason}"
            )


        # ====================================================
        # 条件
        # ====================================================

        with st.expander(
            "現在のエントリー条件を見る"
        ):

            st.write(
                "AI上昇確率：55%以上が必須"
            )

            st.write(
                "AI上昇確率60%以上：強い評価"
            )

            st.write(
                "AI上昇確率70%以上：さらに高い評価"
            )

            st.write(
                "総合スコア：6点以上が必須"
            )

            st.write(
                "RSI適正範囲：40〜70"
            )

            st.write(
                "出来高比率：1.0倍以上で加点"
            )

            st.write(
                "移動平均：株価・5日線・25日線・75日線を評価"
            )

            st.write(
                "市場環境：日経平均・NASDAQ・SOXX・ドル円を評価"
            )


        st.success(
            "🎉 AIエントリー判断まで正常に完了しました。"
        )


    except Exception as e:

        st.error(
            "❌ AIエントリー判断中にエラーが発生しました。"
        )

        st.exception(e)


# ============================================================
# 開発状況
# ============================================================

st.divider()

st.subheader(
    "🚧 開発状況"
)


development_status = {

    "株価データ":
        "✅ 完了",

    "市場データ":
        "✅ 完了",

    "ニュース取得":
        "✅ 完了",

    "テクニカル分析":
        "✅ 完了",

    "AI特徴量":
        "✅ 完了",

    "市場データ対応AI":
        "✅ 完了",

    "固定80/20検証":
        "✅ 完了",

    "ウォークフォワード検証":
        "✅ 完了",

    "売買バックテスト":
        "✅ 完了",

    "エントリー判断":
        "✅ 今回追加",

    "売却判断":
        "🔵 次の段階",

    "リスク管理":
        "🔵 未実装",

    "ペーパートレード":
        "🔵 未実装",

    "実注文":
        "⚪ OFF"
}


st.dataframe(
    pd.DataFrame(
        list(
            development_status.items()
        ),
        columns=[
            "機能",
            "状態"
        ]
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 注意
# ============================================================

st.divider()

st.caption(
    "BUY / WAITはプログラム上の検証用シグナルです。"
    "実際の投資判断や利益を保証するものではありません。"
)

st.caption(
    "現在のエントリースコアや55%・6点などの基準値は"
    "初期設定であり、今後バックテストで検証します。"
)

st.caption(
    "現在は検証段階です。実際の売買注文は行いません。"
)
