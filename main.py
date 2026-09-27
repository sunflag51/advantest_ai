# ============================================================
# アドバンテスト AI売買システム
# main.py
#
# 機能
# 1. アドバンテスト株価取得
# 2. 市場データ取得
# 3. テクニカル分析
# 4. AI特徴量作成
# 5. Random Forest AI学習
# 6. 最新AI予測
# 7. 特徴量重要度
# 8. 固定テスト期間の予測確認
# 9. ウォークフォワード検証
#
# ※実際の売買注文は行いません
# ============================================================


import streamlit as st
import pandas as pd


# ============================================================
# 株価データ
# ============================================================

from data.stock_data import get_stock_data


# ============================================================
# 市場データ
# ============================================================

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
# ウォークフォワード検証
# ============================================================

from backtest.engine import (
    WalkForwardBacktest
)


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
    "アドバンテスト＋市場環境＋テクニカル指標を"
    "AIで分析します。"
)

st.caption(
    "現在はAI分析・検証段階です。"
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

    st.success(
        "✅ 株価データ"
    )


with col2:

    st.success(
        "✅ 市場データ"
    )


with col3:

    st.success(
        "✅ テクニカル分析"
    )


with col4:

    st.success(
        "🤖 AI＋検証"
    )


st.divider()


# ============================================================
# AI説明
# ============================================================

with st.expander(
    "🤖 AIとウォークフォワード検証について"
):

    st.write(
        "AIはアドバンテスト自身のテクニカル指標と、"
        "日経平均・NASDAQ・SOXX・ドル円から作成した"
        "市場特徴量を使用します。"
    )

    st.write(
        "ウォークフォワード検証では、"
        "その時点より過去のデータだけでAIを学習し、"
        "その後の期間を予測します。"
    )

    st.write(
        "一定期間進むごとにAIを再学習することで、"
        "固定80/20分割より実運用に近い形で"
        "予測性能を確認します。"
    )


# ============================================================
# 実行ボタン
# ============================================================

st.subheader(
    "🚀 AI分析"
)


run_analysis = st.button(
    "株価・市場・AI分析を実行",
    type="primary",
    use_container_width=True
)


# ============================================================
# 分析開始
# ============================================================

if run_analysis:

    # ========================================================
    # ① アドバンテスト株価
    # ========================================================

    st.header(
        "① アドバンテスト株価"
    )


    try:

        with st.spinner(
            "アドバンテストの株価を取得しています..."
        ):

            stock_data = get_stock_data(
                stock_code=STOCK_CODE,
                period="5y",
                interval="1d"
            )


        if (
            stock_data is None
            or stock_data.empty
        ):

            st.error(
                "株価データを取得できませんでした。"
            )

            st.stop()


        st.success(
            "✅ 株価データ取得成功"
        )


        # ----------------------------------------------------
        # 最新株価
        # ----------------------------------------------------

        latest_stock = (
            stock_data.iloc[-1]
        )


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


        # ----------------------------------------------------
        # 前日比
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 表示
        # ----------------------------------------------------

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


        st.subheader(
            "📈 株価チャート"
        )


        st.line_chart(
            stock_data[
                ["Close"]
            ].tail(250)
        )


    except Exception as e:

        st.error(
            "❌ 株価データ取得中にエラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ② 市場データ
    # ========================================================

    st.divider()

    st.header(
        "② 市場環境"
    )


    try:

        with st.spinner(
            "日経平均・NASDAQ・SOXX・ドル円を取得しています..."
        ):

            market_data = (
                get_all_market_data(
                    period="5y",
                    interval="1d"
                )
            )


            latest_market = (
                get_latest_market_values(
                    period="5d"
                )
            )


        st.success(
            "✅ 市場データ取得成功"
        )


        # ----------------------------------------------------
        # 最新市場値
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            nikkei = latest_market.get(
                "日経平均"
            )


            if (
                nikkei
                and nikkei.get("close") is not None
            ):

                st.metric(
                    "🇯🇵 日経平均",
                    f"{nikkei['close']:,.2f}"
                )

            else:

                st.warning(
                    "日経平均データなし"
                )


        with col2:

            nasdaq = latest_market.get(
                "NASDAQ"
            )


            if (
                nasdaq
                and nasdaq.get("close") is not None
            ):

                st.metric(
                    "🇺🇸 NASDAQ",
                    f"{nasdaq['close']:,.2f}"
                )

            else:

                st.warning(
                    "NASDAQデータなし"
                )


        col1, col2 = st.columns(2)


        with col1:

            soxx = latest_market.get(
                "SOXX"
            )


            if (
                soxx
                and soxx.get("close") is not None
            ):

                st.metric(
                    "💻 SOXX",
                    f"{soxx['close']:,.2f}"
                )

            else:

                st.warning(
                    "SOXXデータなし"
                )


        with col2:

            usd_jpy = latest_market.get(
                "ドル円"
            )


            if (
                usd_jpy
                and usd_jpy.get("close") is not None
            ):

                st.metric(
                    "💴 ドル円",
                    f"{usd_jpy['close']:,.2f} 円"
                )

            else:

                st.warning(
                    "ドル円データなし"
                )


        # ----------------------------------------------------
        # 市場比較チャート
        # ----------------------------------------------------

        market_chart = pd.DataFrame()


        for market_name in [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円"
        ]:

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


            series = (
                pd.to_numeric(
                    market_df["Close"],
                    errors="coerce"
                )
                .dropna()
            )


            if series.empty:

                continue


            normalized = (
                series
                / series.iloc[0]
                * 100
            )


            normalized.name = (
                market_name
            )


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
                "各市場の表示開始時点を100として比較しています。"
            )

            st.line_chart(
                market_chart.tail(250)
            )


    except Exception as e:

        st.error(
            "❌ 市場データ取得中にエラーが発生しました。"
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

        with st.spinner(
            "テクニカル指標を計算しています..."
        ):

            technical_data = (
                add_all_indicators(
                    stock_data
                )
            )


        latest_indicators = (
            get_latest_indicators(
                technical_data
            )
        )


        st.success(
            "✅ テクニカル分析完了"
        )


        # ----------------------------------------------------
        # 移動平均
        # ----------------------------------------------------

        st.subheader(
            "📊 移動平均"
        )


        sma5 = latest_indicators.get(
            "SMA_5"
        )

        sma25 = latest_indicators.get(
            "SMA_25"
        )

        sma75 = latest_indicators.get(
            "SMA_75"
        )


        col1, col2, col3 = st.columns(3)


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
            "Close",
            "SMA_5",
            "SMA_25",
            "SMA_75"
        ]


        existing_ma_columns = [

            column

            for column in ma_columns

            if column in technical_data.columns
        ]


        st.line_chart(
            technical_data[
                existing_ma_columns
            ].tail(250)
        )


        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        st.subheader(
            "📈 RSI"
        )


        rsi = latest_indicators.get(
            "RSI_14"
        )


        if pd.notna(rsi):

            st.metric(
                "RSI（14日）",
                f"{rsi:.2f}"
            )


        if "RSI_14" in technical_data.columns:

            st.line_chart(
                technical_data[
                    ["RSI_14"]
                ].tail(250)
            )


        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        st.subheader(
            "📉 MACD"
        )


        macd = latest_indicators.get(
            "MACD"
        )

        macd_signal = latest_indicators.get(
            "MACD_Signal"
        )

        macd_histogram = latest_indicators.get(
            "MACD_Histogram"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if pd.notna(macd):

                st.metric(
                    "MACD",
                    f"{macd:.2f}"
                )


        with col2:

            if pd.notna(macd_signal):

                st.metric(
                    "シグナル",
                    f"{macd_signal:.2f}"
                )


        with col3:

            if pd.notna(macd_histogram):

                st.metric(
                    "ヒストグラム",
                    f"{macd_histogram:.2f}"
                )


        if all(
            column in technical_data.columns
            for column in [
                "MACD",
                "MACD_Signal"
            ]
        ):

            st.line_chart(
                technical_data[
                    [
                        "MACD",
                        "MACD_Signal"
                    ]
                ].tail(250)
            )


        # ----------------------------------------------------
        # ボリンジャーバンド
        # ----------------------------------------------------

        st.subheader(
            "📐 ボリンジャーバンド"
        )


        bb_columns = [
            "Close",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower"
        ]


        existing_bb_columns = [

            column

            for column in bb_columns

            if column in technical_data.columns
        ]


        if existing_bb_columns:

            st.line_chart(
                technical_data[
                    existing_bb_columns
                ].tail(250)
            )


        # ----------------------------------------------------
        # ATR / 出来高 / ボラティリティ
        # ----------------------------------------------------

        atr = latest_indicators.get(
            "ATR_14"
        )

        volume_ratio = (
            latest_indicators.get(
                "Volume_Ratio"
            )
        )

        volatility = (
            latest_indicators.get(
                "Volatility_20D"
            )
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if pd.notna(atr):

                st.metric(
                    "ATR（14日）",
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
            "❌ テクニカル分析中にエラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ④ AI特徴量
    # ========================================================

    st.divider()

    st.header(
        "④ 🧠 AI特徴量作成"
    )


    try:

        with st.spinner(
            "アドバンテストと市場データを統合しています..."
        ):

            ai_data = (
                build_ai_features(
                    technical_data=technical_data,
                    market_data=market_data
                )
            )


        ai_feature_columns = (
            get_ai_feature_columns(
                ai_data
            )
        )


        feature_summary = (
            get_feature_summary(
                ai_data
            )
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
                "AI特徴量数",
                f"{len(ai_feature_columns)}"
            )


        with col2:

            st.metric(
                "データ行数",
                f"{feature_summary.get('rows', 0):,}"
            )


        with col3:

            st.metric(
                "欠損値数",
                f"{feature_summary.get('missing_values', 0):,}"
            )


        # ----------------------------------------------------
        # 市場特徴量状態
        # ----------------------------------------------------

        st.subheader(
            "🌏 AI市場特徴量"
        )


        market_status_rows = []


        for market_name in [
            "日経平均",
            "NASDAQ",
            "SOXX",
            "ドル円"
        ]:

            status = (
                market_feature_status.get(
                    market_name,
                    {}
                )
            )


            available = status.get(
                "available",
                False
            )


            feature_count = status.get(
                "feature_count",
                0
            )


            market_status_rows.append(
                {
                    "市場":
                        market_name,

                    "状態":
                        (
                            "✅ 使用"
                            if available
                            else "⚠️ 未使用"
                        ),

                    "特徴量数":
                        feature_count
                }
            )


        st.dataframe(
            pd.DataFrame(
                market_status_rows
            ),
            use_container_width=True,
            hide_index=True
        )


        with st.expander(
            "AIが使用する特徴量を見る"
        ):

            st.dataframe(
                pd.DataFrame(
                    {
                        "特徴量":
                            ai_feature_columns
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "❌ AI特徴量作成中にエラーが発生しました。"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ⑤ AIモデル
    # ========================================================

    st.divider()

    st.header(
        "⑤ 🤖 市場データ対応AI"
    )


    try:

        ai_model = (
            StockPredictionModel()
        )


        with st.spinner(
            "市場データを含めてAIを学習しています..."
        ):

            metrics = (
                ai_model.train(
                    data=ai_data,
                    feature_columns=ai_feature_columns,
                    train_ratio=0.8
                )
            )


        st.success(
            "✅ AI学習完了"
        )


        # ----------------------------------------------------
        # AI学習情報
        # ----------------------------------------------------

        st.subheader(
            "🧠 AI学習情報"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "使用特徴量",
                f"{metrics.get('feature_count', 0)} 個"
            )


        with col2:

            st.metric(
                "学習データ",
                f"{metrics.get('train_samples', 0):,} 件"
            )


        with col3:

            st.metric(
                "テストデータ",
                f"{metrics.get('test_samples', 0):,} 件"
            )


        # ----------------------------------------------------
        # 固定テスト評価
        # ----------------------------------------------------

        st.subheader(
            "📊 固定80/20検証"
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

            fixed_auc = metrics.get(
                "auc"
            )


            if fixed_auc is not None:

                st.metric(
                    "AUC",
                    f"{fixed_auc:.3f}"
                )

            else:

                st.metric(
                    "AUC",
                    "計算不可"
                )


        # ----------------------------------------------------
        # 混同行列
        # ----------------------------------------------------

        confusion_df = pd.DataFrame(
            [
                {
                    "項目": "下落を正しく予測",
                    "件数": metrics.get(
                        "true_negative",
                        0
                    )
                },
                {
                    "項目": "上昇と誤予測",
                    "件数": metrics.get(
                        "false_positive",
                        0
                    )
                },
                {
                    "項目": "下落と誤予測",
                    "件数": metrics.get(
                        "false_negative",
                        0
                    )
                },
                {
                    "項目": "上昇を正しく予測",
                    "件数": metrics.get(
                        "true_positive",
                        0
                    )
                }
            ]
        )


        st.dataframe(
            confusion_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # ⑥ 最新AI予測
        # ====================================================

        st.divider()

        st.header(
            "⑥ 🔮 最新AI予測"
        )


        prediction = (
            ai_model.predict(
                data=ai_data,
                threshold=0.50
            )
        )


        probability_up = (
            prediction[
                "probability_up"
            ]
        )


        probability_down = (
            prediction[
                "probability_down"
            ]
        )


        prediction_text = (
            prediction[
                "prediction_text"
            ]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📈 翌営業日の上昇確率",
                f"{probability_up * 100:.2f}%"
            )


        with col2:

            st.metric(
                "📉 翌営業日の下落確率",
                f"{probability_down * 100:.2f}%"
            )


        st.write(
            "上昇確率"
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


        if prediction_text == "上昇":

            st.success(
                "AI方向予測：上昇側"
            )

        else:

            st.warning(
                "AI方向予測：下落側"
            )


        st.caption(
            "50%は方向分類の境界です。"
            "現段階では売買シグナルではありません。"
        )


        # ====================================================
        # ⑦ 特徴量重要度
        # ====================================================

        st.divider()

        st.header(
            "⑦ 🔍 AIが重視した特徴量"
        )


        importance = (
            ai_model
            .get_feature_importance()
        )


        importance_display = (
            importance.copy()
        )


        importance_display[
            "importance"
        ] = (
            importance_display[
                "importance"
            ]
            * 100
        )


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
            .copy()
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


        # ====================================================
        # ⑧ 固定テスト期間結果
        # ====================================================

        st.divider()

        st.header(
            "⑧ 📋 固定テスト期間の予測"
        )


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


            display_test[
                "正解"
            ] = (
                display_test[
                    "Actual"
                ]
                ==
                display_test[
                    "Prediction"
                ]
            )


            display_test[
                "正解"
            ] = (
                display_test[
                    "正解"
                ]
                .map(
                    {
                        True: "○",
                        False: "×"
                    }
                )
            )


            st.dataframe(
                display_test[
                    [
                        "実際",
                        "AI予測",
                        "上昇確率（%）",
                        "正解"
                    ]
                ].tail(50),
                use_container_width=True
            )


        # ====================================================
        # ⑨ ウォークフォワード検証
        # ====================================================

        st.divider()

        st.header(
            "⑨ 🔄 ウォークフォワード検証"
        )


        st.write(
            "過去のデータだけでAIを学習し、"
            "その後の期間を予測する処理を"
            "時間を進めながら繰り返します。"
        )


        st.caption(
            "初期学習500営業日・20営業日ごとに再学習・"
            "判定基準50%で検証します。"
        )


        try:

            with st.spinner(
                "ウォークフォワード検証を実行しています..."
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


            # ================================================
            # 基本情報
            # ================================================

            st.subheader(
                "🧪 検証情報"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "検証営業日",
                    f"{walk_metrics.get('samples', 0):,} 日"
                )


            with col2:

                st.metric(
                    "AI再学習回数",
                    f"{walk_metrics.get('model_count', 0):,} 回"
                )


            with col3:

                st.metric(
                    "判定基準",
                    f"{walk_metrics.get('threshold', 0.5) * 100:.0f}%"
                )


            # ================================================
            # 評価指標
            # ================================================

            st.subheader(
                "📊 ウォークフォワードAI評価"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Accuracy",
                    f"{walk_metrics.get('accuracy', 0) * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Precision",
                    f"{walk_metrics.get('precision', 0) * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "Recall",
                    f"{walk_metrics.get('recall', 0) * 100:.2f}%"
                )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "F1",
                    f"{walk_metrics.get('f1', 0) * 100:.2f}%"
                )


            with col2:

                walk_auc = (
                    walk_metrics.get(
                        "auc"
                    )
                )


                if walk_auc is not None:

                    st.metric(
                        "AUC",
                        f"{walk_auc:.3f}"
                    )

                else:

                    st.metric(
                        "AUC",
                        "計算不可"
                    )


            # ================================================
            # 上昇割合
            # ================================================

            st.subheader(
                "📈 予測傾向"
            )


            actual_up_rate = (
                walk_metrics.get(
                    "actual_up_rate",
                    0
                )
            )


            predicted_up_rate = (
                walk_metrics.get(
                    "predicted_up_rate",
                    0
                )
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "実際の上昇日割合",
                    f"{actual_up_rate * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "AIが上昇と予測した割合",
                    f"{predicted_up_rate * 100:.2f}%"
                )


            # ================================================
            # 予測方向別リターン
            # ================================================

            st.subheader(
                "💹 予測方向と翌日リターン"
            )


            average_return_when_up = (
                walk_metrics.get(
                    "average_return_when_up",
                    0
                )
            )


            average_return_when_down = (
                walk_metrics.get(
                    "average_return_when_down",
                    0
                )
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "上昇予測日の平均翌日リターン",
                    f"{average_return_when_up * 100:+.3f}%"
                )


            with col2:

                st.metric(
                    "下落予測日の平均翌日リターン",
                    f"{average_return_when_down * 100:+.3f}%"
                )


            # ================================================
            # 60%以上の高確率予測
            # ================================================

            st.subheader(
                "🎯 上昇確率60%以上"
            )


            high_samples = (
                walk_metrics.get(
                    "high_confidence_samples",
                    0
                )
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


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "予測回数",
                    f"{high_samples:,} 回"
                )


            with col2:

                if high_accuracy is not None:

                    st.metric(
                        "正解率",
                        f"{high_accuracy * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "正解率",
                        "データなし"
                    )


            with col3:

                if high_return is not None:

                    st.metric(
                        "平均翌日リターン",
                        f"{high_return * 100:+.3f}%"
                    )

                else:

                    st.metric(
                        "平均翌日リターン",
                        "データなし"
                    )


            # ================================================
            # 混同行列
            # ================================================

            st.subheader(
                "📋 ウォークフォワード予測内訳"
            )


            walk_confusion_df = (
                pd.DataFrame(
                    [
                        {
                            "項目":
                                "下落を正しく予測",

                            "件数":
                                walk_metrics.get(
                                    "true_negative",
                                    0
                                )
                        },

                        {
                            "項目":
                                "上昇と誤予測",

                            "件数":
                                walk_metrics.get(
                                    "false_positive",
                                    0
                                )
                        },

                        {
                            "項目":
                                "下落と誤予測",

                            "件数":
                                walk_metrics.get(
                                    "false_negative",
                                    0
                                )
                        },

                        {
                            "項目":
                                "上昇を正しく予測",

                            "件数":
                                walk_metrics.get(
                                    "true_positive",
                                    0
                                )
                        }
                    ]
                )
            )


            st.dataframe(
                walk_confusion_df,
                use_container_width=True,
                hide_index=True
            )


            # ================================================
            # 上昇確率チャート
            # ================================================

            st.subheader(
                "📈 AI上昇確率の推移"
            )


            probability_chart = (
                walk_results[
                    [
                        "Probability_Up"
                    ]
                ].copy()
            )


            probability_chart[
                "Probability_Up"
            ] = (
                probability_chart[
                    "Probability_Up"
                ]
                * 100
            )


            probability_chart.rename(
                columns={
                    "Probability_Up":
                        "上昇確率（%）"
                },
                inplace=True
            )


            st.line_chart(
                probability_chart
            )


            # ================================================
            # 累積翌日リターン参考表示
            #
            # Prediction == 1 の日の翌日リターンを
            # 単純累積した参考値
            #
            # まだ正式な売買バックテストではない
            # ================================================

            strategy_reference = (
                walk_results.copy()
            )


            strategy_reference[
                "AI参考リターン"
            ] = (
                strategy_reference[
                    "Next_Return"
                ]
                *
                strategy_reference[
                    "Prediction"
                ]
            )


            strategy_reference[
                "AI参考資産指数"
            ] = (
                (
                    1
                    +
                    strategy_reference[
                        "AI参考リターン"
                    ]
                )
                .cumprod()
                * 100
            )


            strategy_reference[
                "BuyHold参考指数"
            ] = (
                (
                    1
                    +
                    strategy_reference[
                        "Next_Return"
                    ]
                )
                .cumprod()
                * 100
            )


            st.subheader(
                "📊 参考：AI予測とBuy & Hold"
            )


            st.caption(
                "これは売買手数料・スリッページ・"
                "翌日寄り付き約定などをまだ考慮していない"
                "参考表示です。正式な売買バックテストではありません。"
            )


            st.line_chart(
                strategy_reference[
                    [
                        "AI参考資産指数",
                        "BuyHold参考指数"
                    ]
                ]
            )


            # ================================================
            # 最新50件
            # ================================================

            with st.expander(
                "ウォークフォワード予測の最新50件を見る"
            ):

                walk_display = (
                    walk_results.copy()
                )


                walk_display[
                    "上昇確率（%）"
                ] = (
                    walk_display[
                        "Probability_Up"
                    ]
                    * 100
                )


                walk_display[
                    "翌日騰落率（%）"
                ] = (
                    walk_display[
                        "Next_Return"
                    ]
                    * 100
                )


                walk_display[
                    "実際"
                ] = (
                    walk_display[
                        "Actual"
                    ]
                    .map(
                        {
                            1: "上昇",
                            0: "下落"
                        }
                    )
                )


                walk_display[
                    "AI予測"
                ] = (
                    walk_display[
                        "Prediction"
                    ]
                    .map(
                        {
                            1: "上昇",
                            0: "下落"
                        }
                    )
                )


                walk_display[
                    "正解"
                ] = (
                    walk_display[
                        "Correct"
                    ]
                    .map(
                        {
                            1: "○",
                            0: "×"
                        }
                    )
                )


                st.dataframe(
                    walk_display[
                        [
                            "Close",
                            "Next_Close",
                            "実際",
                            "AI予測",
                            "上昇確率（%）",
                            "翌日騰落率（%）",
                            "正解"
                        ]
                    ].tail(50),
                    use_container_width=True
                )


            # ================================================
            # 再学習履歴
            # ================================================

            with st.expander(
                "AI再学習履歴を見る"
            ):

                training_log = (
                    walk_forward
                    .get_training_log()
                )


                st.dataframe(
                    training_log,
                    use_container_width=True,
                    hide_index=True
                )


            st.success(
                "🎉 ウォークフォワード検証まで正常に完了しました。"
            )


        except Exception as e:

            st.error(
                "❌ ウォークフォワード検証中にエラーが発生しました。"
            )

            st.exception(e)


    except Exception as e:

        st.error(
            "❌ AIモデル実行中にエラーが発生しました。"
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

    "Streamlit画面":
        "✅ 完了",

    "アドバンテスト株価":
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
        "✅ 今回追加",

    "正式な売買バックテスト":
        "🔵 次の段階",

    "エントリー判断":
        "🔵 未実装",

    "売却判断":
        "🔵 未実装",

    "リスク管理":
        "🔵 未実装",

    "ペーパートレード":
        "🔵 未実装",

    "自動売買":
        "⚪ OFF"
}


status_df = pd.DataFrame(
    list(
        development_status.items()
    ),
    columns=[
        "機能",
        "状態"
    ]
)


st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 注意
# ============================================================

st.divider()

st.caption(
    "AI予測は過去データを使った統計的予測であり、"
    "将来の株価上昇や利益を保証するものではありません。"
)

st.caption(
    "ウォークフォワード検証結果も、"
    "実際の取引結果を保証するものではありません。"
)

st.caption(
    "現在は検証段階です。"
    "実際の売買注文は一切行いません。"
)
