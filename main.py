# ============================================================
# アドバンテスト AI売買システム
# main.py
#
# 現在の機能
# 1. アドバンテスト株価取得
# 2. 市場環境データ取得
# 3. テクニカル指標計算
# 4. AIモデル学習
# 5. AIモデル検証
# 6. 翌営業日の上昇確率予測
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
# AIモデル
# ============================================================

from ai.model import (
    StockPredictionModel
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
    "株価・市場環境・テクニカル分析・AI予測"
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

    st.info(
        "🤖 AI予測"
    )


st.divider()


# ============================================================
# 実行ボタン
# ============================================================

st.subheader(
    "🚀 AI分析"
)


run_analysis = st.button(
    "株価・市場・AI分析を実行",
    type="primary"
)


# ============================================================
# 分析実行
# ============================================================

if run_analysis:

    # ========================================================
    # ① 株価データ取得
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

        latest = stock_data.iloc[-1]


        close_price = float(
            latest["Close"]
        )

        open_price = float(
            latest["Open"]
        )

        high_price = float(
            latest["High"]
        )

        low_price = float(
            latest["Low"]
        )

        volume = int(
            latest["Volume"]
        )


        # ----------------------------------------------------
        # 株価表示
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:

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
                "高値",
                f"{high_price:,.0f} 円"
            )


        with col4:

            st.metric(
                "安値",
                f"{low_price:,.0f} 円"
            )


        with col5:

            st.metric(
                "出来高",
                f"{volume:,}"
            )


        # ----------------------------------------------------
        # 株価チャート
        # ----------------------------------------------------

        st.subheader(
            "アドバンテスト終値"
        )


        st.line_chart(
            stock_data[
                ["Close"]
            ]
        )


    except Exception as e:

        st.error(
            "❌ 株価データ取得エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ② 市場環境
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


            latest_market = (
                get_latest_market_values(
                    period="5d"
                )
            )


        st.success(
            "✅ 市場データ取得成功"
        )


        # ----------------------------------------------------
        # 日経平均
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


        # ----------------------------------------------------
        # NASDAQ
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SOXX
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ドル円
        # ----------------------------------------------------

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


    except Exception as e:

        st.error(
            "❌ 市場データ取得エラー"
        )

        st.exception(e)


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

            technical_data = add_all_indicators(
                stock_data
            )


        st.success(
            "✅ テクニカル分析完了"
        )


        latest_indicators = (
            get_latest_indicators(
                technical_data
            )
        )


        # ----------------------------------------------------
        # 移動平均
        # ----------------------------------------------------

        st.subheader(
            "📊 移動平均"
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


        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi = latest_indicators.get(
            "RSI_14"
        )


        st.subheader(
            "📈 RSI"
        )


        if pd.notna(rsi):

            st.metric(
                "RSI（14日）",
                f"{rsi:.2f}"
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


        # ----------------------------------------------------
        # ボリンジャーバンド
        # ----------------------------------------------------

        st.subheader(
            "📐 ボリンジャーバンド"
        )


        bb_chart = technical_data[
            [
                "Close",
                "BB_Upper",
                "BB_Middle",
                "BB_Lower"
            ]
        ].tail(250)


        st.line_chart(
            bb_chart
        )


        # ----------------------------------------------------
        # その他
        # ----------------------------------------------------

        st.subheader(
            "📊 その他の指標"
        )


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
            "❌ テクニカル分析エラー"
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # ④ AIモデル
    # ========================================================

    st.divider()

    st.header(
        "④ 🤖 AI予測"
    )


    st.write(
        "過去のアドバンテスト株価・テクニカル指標を使用して、"
        "翌営業日の上昇確率を予測します。"
    )


    try:

        # ----------------------------------------------------
        # AIモデル作成
        # ----------------------------------------------------

        ai_model = StockPredictionModel()


        # ----------------------------------------------------
        # AI学習
        # ----------------------------------------------------

        with st.spinner(
            "AIモデルを学習しています..."
        ):

            metrics = ai_model.train(
                data=technical_data,
                train_ratio=0.8
            )


        st.success(
            "✅ AI学習完了"
        )


        # ----------------------------------------------------
        # AI評価
        # ----------------------------------------------------

        st.subheader(
            "📊 AIモデル検証結果"
        )


        col1, col2, col3, col4 = st.columns(4)


        accuracy = metrics.get(
            "accuracy"
        )

        precision = metrics.get(
            "precision"
        )

        recall = metrics.get(
            "recall"
        )

        auc = metrics.get(
            "auc"
        )


        with col1:

            if accuracy is not None:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


        with col2:

            if precision is not None:

                st.metric(
                    "Precision",
                    f"{precision * 100:.2f}%"
                )


        with col3:

            if recall is not None:

                st.metric(
                    "Recall",
                    f"{recall * 100:.2f}%"
                )


        with col4:

            if auc is not None:

                st.metric(
                    "AUC",
                    f"{auc:.3f}"
                )


        # ----------------------------------------------------
        # 学習件数
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "学習データ件数",
                f"{metrics['train_samples']:,}"
            )


        with col2:

            st.metric(
                "テストデータ件数",
                f"{metrics['test_samples']:,}"
            )


        # ====================================================
        # 最新予測
        # ====================================================

        st.divider()

        st.subheader(
            "🔮 最新AI予測"
        )


        with st.spinner(
            "最新データから上昇確率を予測しています..."
        ):

            prediction = ai_model.predict(
                technical_data
            )


        probability_up = (
            prediction["probability_up"]
        )


        probability_down = (
            prediction["probability_down"]
        )


        prediction_text = (
            prediction["prediction_text"]
        )


        # ----------------------------------------------------
        # 大きく表示
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # AI予測
        # ----------------------------------------------------

        if prediction_text == "上昇":

            st.success(
                "AI予測：翌営業日は上昇側"
            )

        else:

            st.warning(
                "AI予測：翌営業日は下落側"
            )


        # ----------------------------------------------------
        # 確率バー
        # ----------------------------------------------------

        st.write(
            "上昇確率"
        )


        st.progress(
            probability_up
        )


        # ====================================================
        # 特徴量重要度
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 AIが重視した特徴量"
        )


        importance = (
            ai_model.get_feature_importance()
        )


        importance_display = (
            importance.copy()
        )


        importance_display[
            "importance"
        ] = (
            importance_display[
                "importance"
            ] * 100
        )


        importance_display.rename(
            columns={
                "feature": "特徴量",
                "importance": "重要度（%）"
            },
            inplace=True
        )


        st.dataframe(
            importance_display,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # 特徴量グラフ
        # ----------------------------------------------------

        importance_chart = (
            importance_display
            .set_index("特徴量")
            [["重要度（%）"]]
        )


        st.bar_chart(
            importance_chart
        )


        # ====================================================
        # 最新テクニカルデータ
        # ====================================================

        with st.expander(
            "最新20営業日のデータを見る"
        ):

            st.dataframe(
                technical_data.tail(20),
                use_container_width=True
            )


        # ====================================================
        # 完了
        # ====================================================

        st.success(
            "🎉 AI予測まで正常に完了しました。"
        )


    except Exception as e:

        st.error(
            "❌ AIモデルの実行中にエラーが発生しました。"
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

    "アドバンテスト株価取得":
        "✅ 完了",

    "市場データ取得":
        "✅ 完了",

    "ニュース取得":
        "✅ 完了",

    "テクニカル分析":
        "✅ 完了",

    "AI予測モデル":
        "✅ 今回追加",

    "AIバックテスト":
        "🔵 次の段階",

    "エントリー判断":
        "🔵 未実装",

    "売却判断":
        "🔵 未実装",

    "リスク管理":
        "🔵 未実装",

    "仮想売買":
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
# 注意事項
# ============================================================

st.divider()

st.caption(
    "AIの予測値は過去データに基づく統計的な予測であり、"
    "将来の株価を保証するものではありません。"
)

st.caption(
    "現在は分析・検証段階であり、実際の売買注文は行いません。"
)
