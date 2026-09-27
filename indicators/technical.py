# ============================================================
# アドバンテスト AI売買システム
# indicators/technical.py
#
# テクニカル指標計算プログラム
#
# このファイルの役割
# ・移動平均線
# ・EMA
# ・RSI
# ・MACD
# ・ボリンジャーバンド
# ・ATR
# ・出来高移動平均
# ・価格変化率
# ・ボラティリティ
#
# を計算する
#
# ※このファイルでは売買判断を行いません
# ============================================================


import pandas as pd
import numpy as np


# ============================================================
# 移動平均線
# ============================================================

def add_moving_averages(
    data,
    short_period=5,
    medium_period=25,
    long_period=75
):
    """
    SMA（単純移動平均線）を追加する
    """

    df = data.copy()

    df[f"SMA_{short_period}"] = (
        df["Close"]
        .rolling(short_period)
        .mean()
    )

    df[f"SMA_{medium_period}"] = (
        df["Close"]
        .rolling(medium_period)
        .mean()
    )

    df[f"SMA_{long_period}"] = (
        df["Close"]
        .rolling(long_period)
        .mean()
    )

    return df


# ============================================================
# EMA
# ============================================================

def add_ema(
    data,
    short_period=12,
    long_period=26
):
    """
    指数移動平均線を追加する
    """

    df = data.copy()

    df[f"EMA_{short_period}"] = (
        df["Close"]
        .ewm(
            span=short_period,
            adjust=False
        )
        .mean()
    )

    df[f"EMA_{long_period}"] = (
        df["Close"]
        .ewm(
            span=long_period,
            adjust=False
        )
        .mean()
    )

    return df


# ============================================================
# RSI
# ============================================================

def add_rsi(
    data,
    period=14
):
    """
    RSIを追加する

    RSI
    70以上 → 買われすぎの目安
    30以下 → 売られすぎの目安

    ※AIでは単独で売買判断には使用しません
    """

    df = data.copy()

    delta = df["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    rs = (
        average_gain /
        average_loss.replace(
            0,
            np.nan
        )
    )

    df[f"RSI_{period}"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return df


# ============================================================
# MACD
# ============================================================

def add_macd(
    data,
    fast_period=12,
    slow_period=26,
    signal_period=9
):
    """
    MACDを追加する
    """

    df = data.copy()

    ema_fast = (
        df["Close"]
        .ewm(
            span=fast_period,
            adjust=False
        )
        .mean()
    )

    ema_slow = (
        df["Close"]
        .ewm(
            span=slow_period,
            adjust=False
        )
        .mean()
    )

    macd = (
        ema_fast -
        ema_slow
    )

    signal = (
        macd
        .ewm(
            span=signal_period,
            adjust=False
        )
        .mean()
    )

    histogram = (
        macd -
        signal
    )

    df["MACD"] = macd

    df["MACD_Signal"] = signal

    df["MACD_Histogram"] = histogram

    return df


# ============================================================
# ボリンジャーバンド
# ============================================================

def add_bollinger_bands(
    data,
    period=20,
    std_multiplier=2
):
    """
    ボリンジャーバンドを追加する
    """

    df = data.copy()

    middle = (
        df["Close"]
        .rolling(period)
        .mean()
    )

    std = (
        df["Close"]
        .rolling(period)
        .std()
    )

    upper = (
        middle +
        std_multiplier * std
    )

    lower = (
        middle -
        std_multiplier * std
    )

    df["BB_Middle"] = middle

    df["BB_Upper"] = upper

    df["BB_Lower"] = lower

    # --------------------------------------------------------
    # バンド幅
    # --------------------------------------------------------

    df["BB_Width"] = (
        (
            upper - lower
        ) / middle
    )

    # --------------------------------------------------------
    # 現在価格がバンドのどの位置にあるか
    # --------------------------------------------------------

    band_range = (
        upper - lower
    )

    df["BB_Position"] = (
        (
            df["Close"] - lower
        ) /
        band_range.replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# ATR
# ============================================================

def add_atr(
    data,
    period=14
):
    """
    ATR（Average True Range）を追加する

    値動きの大きさを表す指標
    """

    df = data.copy()

    previous_close = (
        df["Close"].shift(1)
    )

    high_low = (
        df["High"] -
        df["Low"]
    )

    high_previous = (
        (
            df["High"] -
            previous_close
        ).abs()
    )

    low_previous = (
        (
            df["Low"] -
            previous_close
        ).abs()
    )

    true_range = pd.concat(
        [
            high_low,
            high_previous,
            low_previous
        ],
        axis=1
    ).max(
        axis=1
    )

    df[f"ATR_{period}"] = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    return df


# ============================================================
# 出来高指標
# ============================================================

def add_volume_indicators(
    data,
    period=20
):
    """
    出来高関連の指標を追加する
    """

    df = data.copy()

    df[f"Volume_SMA_{period}"] = (
        df["Volume"]
        .rolling(period)
        .mean()
    )

    # --------------------------------------------------------
    # 現在出来高 / 平均出来高
    # --------------------------------------------------------

    df["Volume_Ratio"] = (
        df["Volume"] /
        df[f"Volume_SMA_{period}"].replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# 価格変化率
# ============================================================

def add_price_returns(
    data
):
    """
    価格の変化率を追加する
    """

    df = data.copy()

    # 前日比
    df["Return_1D"] = (
        df["Close"]
        .pct_change()
    )

    # 5営業日前比
    df["Return_5D"] = (
        df["Close"]
        .pct_change(5)
    )

    # 20営業日前比
    df["Return_20D"] = (
        df["Close"]
        .pct_change(20)
    )

    # 60営業日前比
    df["Return_60D"] = (
        df["Close"]
        .pct_change(60)
    )

    return df


# ============================================================
# ボラティリティ
# ============================================================

def add_volatility(
    data,
    period=20
):
    """
    過去20営業日の価格変動率から
    年率換算ボラティリティを計算する
    """

    df = data.copy()

    daily_return = (
        df["Close"]
        .pct_change()
    )

    df[f"Volatility_{period}D"] = (
        daily_return
        .rolling(period)
        .std()
        * np.sqrt(252)
    )

    return df


# ============================================================
# トレンド情報
# ============================================================

def add_trend_features(
    data
):
    """
    移動平均線を利用したトレンド情報を追加する
    """

    df = data.copy()

    # --------------------------------------------------------
    # 5日線と25日線
    # --------------------------------------------------------

    if (
        "SMA_5" in df.columns
        and "SMA_25" in df.columns
    ):

        df["Trend_5_25"] = (
            df["SMA_5"] /
            df["SMA_25"] -
            1
        )


    # --------------------------------------------------------
    # 25日線と75日線
    # --------------------------------------------------------

    if (
        "SMA_25" in df.columns
        and "SMA_75" in df.columns
    ):

        df["Trend_25_75"] = (
            df["SMA_25"] /
            df["SMA_75"] -
            1
        )


    # --------------------------------------------------------
    # 株価と25日線の位置
    # --------------------------------------------------------

    if "SMA_25" in df.columns:

        df["Price_vs_SMA25"] = (
            df["Close"] /
            df["SMA_25"] -
            1
        )


    # --------------------------------------------------------
    # 株価と75日線の位置
    # --------------------------------------------------------

    if "SMA_75" in df.columns:

        df["Price_vs_SMA75"] = (
            df["Close"] /
            df["SMA_75"] -
            1
        )


    return df


# ============================================================
# すべてのテクニカル指標を計算
# ============================================================

def add_all_indicators(
    data,
    short_ma=5,
    medium_ma=25,
    long_ma=75,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bollinger_period=20,
    bollinger_std=2,
    atr_period=14,
    volume_period=20,
    volatility_period=20
):
    """
    すべてのテクニカル指標を一括計算する

    AI学習用データを作成するときに
    この関数を使用する
    """

    # --------------------------------------------------------
    # 入力データ確認
    # --------------------------------------------------------

    if data is None:

        raise ValueError(
            "株価データがNoneです。"
        )


    if data.empty:

        raise ValueError(
            "株価データが空です。"
        )


    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]


    if missing_columns:

        raise ValueError(
            "必要な列がありません："
            + ", ".join(missing_columns)
        )


    # --------------------------------------------------------
    # データコピー
    # --------------------------------------------------------

    df = data.copy()


    # --------------------------------------------------------
    # 各指標計算
    # --------------------------------------------------------

    df = add_moving_averages(
        df,
        short_period=short_ma,
        medium_period=medium_ma,
        long_period=long_ma
    )


    df = add_ema(
        df,
        short_period=macd_fast,
        long_period=macd_slow
    )


    df = add_rsi(
        df,
        period=rsi_period
    )


    df = add_macd(
        df,
        fast_period=macd_fast,
        slow_period=macd_slow,
        signal_period=macd_signal
    )


    df = add_bollinger_bands(
        df,
        period=bollinger_period,
        std_multiplier=bollinger_std
    )


    df = add_atr(
        df,
        period=atr_period
    )


    df = add_volume_indicators(
        df,
        period=volume_period
    )


    df = add_price_returns(
        df
    )


    df = add_volatility(
        df,
        period=volatility_period
    )


    df = add_trend_features(
        df
    )


    return df


# ============================================================
# 最新のテクニカル指標を取得
# ============================================================

def get_latest_indicators(
    data
):
    """
    最新日のテクニカル指標を取得する
    """

    if data is None or data.empty:

        return None

    return data.iloc[-1].copy()


# ============================================================
# AI学習用データを作成
# ============================================================

def prepare_ai_features(
    data
):
    """
    AI学習に使用する特徴量を作成する

    注意：
    ここでは将来予測の正解データ（target）は作らない。
    """

    if data is None or data.empty:

        raise ValueError(
            "テクニカルデータがありません。"
        )


    df = data.copy()


    # --------------------------------------------------------
    # AIに使用する候補特徴量
    # --------------------------------------------------------

    feature_columns = [

        "Return_1D",
        "Return_5D",
        "Return_20D",
        "Return_60D",

        "RSI_14",

        "MACD",
        "MACD_Signal",
        "MACD_Histogram",

        "BB_Width",
        "BB_Position",

        "ATR_14",

        "Volume_Ratio",

        "Volatility_20D",

        "Trend_5_25",
        "Trend_25_75",

        "Price_vs_SMA25",
        "Price_vs_SMA75"
    ]


    # --------------------------------------------------------
    # 存在する特徴量だけ使用
    # --------------------------------------------------------

    available_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]


    features = df[
        available_features
    ].copy()


    # --------------------------------------------------------
    # 無限大を欠損値へ
    # --------------------------------------------------------

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    return features


# ============================================================
# テクニカル指標の簡易説明
# ============================================================

def get_indicator_description():
    """
    各指標の説明を返す
    """

    return {

        "SMA":
            "指定期間の単純移動平均",

        "EMA":
            "直近の価格を重視する指数移動平均",

        "RSI":
            "価格の上昇・下落の勢いを測定",

        "MACD":
            "短期・長期EMAの差からトレンド変化を分析",

        "Bollinger Bands":
            "価格の変動範囲とボラティリティを分析",

        "ATR":
            "価格変動の大きさを測定",

        "Volume Ratio":
            "現在の出来高が平均と比べてどの程度かを測定",

        "Return":
            "一定期間の価格変化率",

        "Volatility":
            "価格変動の大きさを測定",

        "Trend":
            "移動平均線の位置関係からトレンドを分析"
    }
