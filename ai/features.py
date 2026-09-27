# ============================================================
# アドバンテスト AI売買システム
# ai/features.py
#
# AI特徴量作成
#
# 目的
# ・アドバンテストのテクニカル指標
# ・日経平均
# ・NASDAQ
# ・SOXX
# ・ドル円
#
# を1つのAI学習用データにまとめる
#
# 重要
# 米国市場（NASDAQ / SOXX）は
# 日本市場との時間差を考慮して1行シフトする
#
# ※未来データをAIに見せないことを最優先する
# ============================================================


import numpy as np
import pandas as pd


# ============================================================
# 市場名
# ============================================================

MARKET_NAMES = [
    "日経平均",
    "NASDAQ",
    "SOXX",
    "ドル円",
]


# ============================================================
# 米国市場
#
# 日本市場との時間差を考慮する対象
# ============================================================

US_MARKETS = [
    "NASDAQ",
    "SOXX",
]


# ============================================================
# 基本ユーティリティ
# ============================================================

def _prepare_index(data):
    """
    インデックスをDatetimeIndexへ統一する。

    タイムゾーン情報がある場合は削除し、
    日付単位で扱えるようにする。
    """

    if data is None:

        return None


    df = data.copy()


    if df.empty:

        return df


    # --------------------------------------------------------
    # DatetimeIndexへ変換
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index,
        errors="coerce"
    )


    # --------------------------------------------------------
    # 不正な日付を削除
    # --------------------------------------------------------

    df = df[
        ~df.index.isna()
    ]


    # --------------------------------------------------------
    # タイムゾーン削除
    # --------------------------------------------------------

    try:

        if df.index.tz is not None:

            df.index = (
                df.index.tz_localize(None)
            )

    except Exception:

        pass


    # --------------------------------------------------------
    # 時刻部分を削除
    # --------------------------------------------------------

    df.index = df.index.normalize()


    # --------------------------------------------------------
    # 重複日付があれば最後を使用
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]


    # --------------------------------------------------------
    # 日付順
    # --------------------------------------------------------

    df.sort_index(
        inplace=True
    )


    return df


# ============================================================
# 市場データから特徴量を作成
# ============================================================

def create_market_features(
    market_df,
    market_name,
    shift_days=0
):
    """
    1つの市場データからAI特徴量を作成する。

    作成する特徴量

    ・1日リターン
    ・5日リターン
    ・20日リターン
    ・5日移動平均との乖離
    ・20日移動平均との乖離
    ・20日ボラティリティ

    shift_daysを指定すると、
    特徴量を後ろへシフトする。
    """

    if market_df is None:

        raise ValueError(
            f"{market_name}のデータがありません。"
        )


    if market_df.empty:

        raise ValueError(
            f"{market_name}のデータが空です。"
        )


    df = _prepare_index(
        market_df
    )


    if "Close" not in df.columns:

        raise ValueError(
            f"{market_name}にClose列がありません。"
        )


    # --------------------------------------------------------
    # Closeを数値へ
    # --------------------------------------------------------

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 特徴量データフレーム
    # --------------------------------------------------------

    features = pd.DataFrame(
        index=df.index
    )


    # --------------------------------------------------------
    # 1日リターン
    # --------------------------------------------------------

    features[
        f"{market_name}_Return_1D"
    ] = close.pct_change(
        fill_method=None
    )


    # --------------------------------------------------------
    # 5日リターン
    # --------------------------------------------------------

    features[
        f"{market_name}_Return_5D"
    ] = close.pct_change(
        periods=5,
        fill_method=None
    )


    # --------------------------------------------------------
    # 20日リターン
    # --------------------------------------------------------

    features[
        f"{market_name}_Return_20D"
    ] = close.pct_change(
        periods=20,
        fill_method=None
    )


    # --------------------------------------------------------
    # 5日移動平均
    # --------------------------------------------------------

    sma5 = close.rolling(
        window=5
    ).mean()


    # --------------------------------------------------------
    # 20日移動平均
    # --------------------------------------------------------

    sma20 = close.rolling(
        window=20
    ).mean()


    # --------------------------------------------------------
    # 移動平均乖離率
    # --------------------------------------------------------

    features[
        f"{market_name}_Price_vs_SMA5"
    ] = (
        close / sma5
        - 1
    )


    features[
        f"{market_name}_Price_vs_SMA20"
    ] = (
        close / sma20
        - 1
    )


    # --------------------------------------------------------
    # 20日ボラティリティ
    # --------------------------------------------------------

    daily_return = close.pct_change(
        fill_method=None
    )


    features[
        f"{market_name}_Volatility_20D"
    ] = (
        daily_return
        .rolling(
            window=20
        )
        .std()
        * np.sqrt(252)
    )


    # --------------------------------------------------------
    # 情報利用タイミング調整
    #
    # NASDAQ / SOXXなど
    # --------------------------------------------------------

    if shift_days > 0:

        features = features.shift(
            shift_days
        )


    # --------------------------------------------------------
    # 無限大処理
    # --------------------------------------------------------

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    return features


# ============================================================
# 市場特徴量をまとめる
# ============================================================

def build_market_feature_table(
    market_data,
    stock_index
):
    """
    全市場データをまとめて
    アドバンテストの日付へ合わせる。

    market_dataは

    {
        "日経平均": DataFrame,
        "NASDAQ": DataFrame,
        "SOXX": DataFrame,
        "ドル円": DataFrame
    }

    の形式を想定。
    """

    if market_data is None:

        raise ValueError(
            "市場データがありません。"
        )


    # --------------------------------------------------------
    # アドバンテストの日付
    # --------------------------------------------------------

    stock_index = pd.to_datetime(
        stock_index
    )


    try:

        if stock_index.tz is not None:

            stock_index = (
                stock_index.tz_localize(None)
            )

    except Exception:

        pass


    stock_index = (
        stock_index.normalize()
    )


    # --------------------------------------------------------
    # 最終市場特徴量
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=stock_index
    )


    # ========================================================
    # 各市場
    # ========================================================

    for market_name in MARKET_NAMES:

        if market_name not in market_data:

            continue


        market_df = market_data[
            market_name
        ]


        if (
            market_df is None
            or market_df.empty
        ):

            continue


        # ----------------------------------------------------
        # 米国市場だけ1行シフト
        # ----------------------------------------------------

        if market_name in US_MARKETS:

            shift_days = 1

        else:

            shift_days = 0


        # ----------------------------------------------------
        # 市場特徴量作成
        # ----------------------------------------------------

        market_features = (
            create_market_features(
                market_df=market_df,
                market_name=market_name,
                shift_days=shift_days
            )
        )


        # ----------------------------------------------------
        # アドバンテスト日付へ結合
        # ----------------------------------------------------

        result = result.join(
            market_features,
            how="left"
        )


    # ========================================================
    # 休場日の違いを処理
    # ========================================================

    result = result.ffill()


    # ========================================================
    # 無限大処理
    # ========================================================

    result.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    return result


# ============================================================
# アドバンテスト特徴量
# ============================================================

def get_stock_feature_columns():
    """
    indicators/technical.py で作成した
    アドバンテスト側の特徴量一覧
    """

    return [

        # ----------------------------------------------------
        # リターン
        # ----------------------------------------------------

        "Return_1D",
        "Return_5D",
        "Return_20D",
        "Return_60D",

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        "RSI_14",

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        "MACD",
        "MACD_Signal",
        "MACD_Histogram",

        # ----------------------------------------------------
        # ボリンジャーバンド
        # ----------------------------------------------------

        "BB_Width",
        "BB_Position",

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        "ATR_14",

        # ----------------------------------------------------
        # 出来高
        # ----------------------------------------------------

        "Volume_Ratio",

        # ----------------------------------------------------
        # ボラティリティ
        # ----------------------------------------------------

        "Volatility_20D",

        # ----------------------------------------------------
        # トレンド
        # ----------------------------------------------------

        "Trend_5_25",
        "Trend_25_75",

        # ----------------------------------------------------
        # 移動平均乖離
        # ----------------------------------------------------

        "Price_vs_SMA25",
        "Price_vs_SMA75",
    ]


# ============================================================
# 市場特徴量名
# ============================================================

def get_market_feature_columns():
    """
    市場側で作成する特徴量名一覧
    """

    columns = []


    for market_name in MARKET_NAMES:

        columns.extend(
            [
                f"{market_name}_Return_1D",
                f"{market_name}_Return_5D",
                f"{market_name}_Return_20D",

                f"{market_name}_Price_vs_SMA5",
                f"{market_name}_Price_vs_SMA20",

                f"{market_name}_Volatility_20D",
            ]
        )


    return columns


# ============================================================
# AI特徴量名一覧
# ============================================================

def get_ai_feature_columns(
    data=None
):
    """
    AIで使用する特徴量一覧を返す。

    dataを指定した場合は、
    実際に存在する列だけを返す。
    """

    columns = (
        get_stock_feature_columns()
        + get_market_feature_columns()
    )


    if data is None:

        return columns


    existing_columns = [
        column
        for column in columns
        if column in data.columns
    ]


    return existing_columns


# ============================================================
# AI特徴量統合
# ============================================================

def build_ai_features(
    technical_data,
    market_data
):
    """
    アドバンテストのテクニカルデータと
    市場データを統合する。

    戻り値にはCloseなど元の株価データも残す。

    そのため、
    ai/model.py のcreate_target()も
    そのまま利用できる。
    """

    if technical_data is None:

        raise ValueError(
            "テクニカルデータがありません。"
        )


    if technical_data.empty:

        raise ValueError(
            "テクニカルデータが空です。"
        )


    # --------------------------------------------------------
    # アドバンテストデータ整理
    # --------------------------------------------------------

    stock_df = _prepare_index(
        technical_data
    )


    # --------------------------------------------------------
    # 市場特徴量
    # --------------------------------------------------------

    market_features = (
        build_market_feature_table(
            market_data=market_data,
            stock_index=stock_df.index
        )
    )


    # --------------------------------------------------------
    # 結合
    # --------------------------------------------------------

    combined = stock_df.join(
        market_features,
        how="left"
    )


    # --------------------------------------------------------
    # 無限大処理
    # --------------------------------------------------------

    combined.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    # --------------------------------------------------------
    # 日付順
    # --------------------------------------------------------

    combined.sort_index(
        inplace=True
    )


    return combined


# ============================================================
# AIモデルへ渡すXを作成
# ============================================================

def prepare_feature_matrix(
    data,
    dropna=True
):
    """
    AIモデルに渡す特徴量Xを作成する。
    """

    if data is None:

        raise ValueError(
            "AI特徴量データがありません。"
        )


    if data.empty:

        raise ValueError(
            "AI特徴量データが空です。"
        )


    # --------------------------------------------------------
    # 使用可能な特徴量
    # --------------------------------------------------------

    feature_columns = (
        get_ai_feature_columns(
            data
        )
    )


    if not feature_columns:

        raise ValueError(
            "AIで使用できる特徴量がありません。"
        )


    # --------------------------------------------------------
    # 特徴量のみ抽出
    # --------------------------------------------------------

    X = data[
        feature_columns
    ].copy()


    # --------------------------------------------------------
    # 数値変換
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # 無限大
    # --------------------------------------------------------

    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    # --------------------------------------------------------
    # 欠損削除
    # --------------------------------------------------------

    if dropna:

        X.dropna(
            inplace=True
        )


    return X


# ============================================================
# AI用データ確認
# ============================================================

def get_feature_summary(
    data
):
    """
    AI特徴量の状態を確認する。
    """

    if data is None or data.empty:

        return {
            "rows": 0,
            "features": 0,
            "missing_values": 0,
        }


    feature_columns = (
        get_ai_feature_columns(
            data
        )
    )


    if not feature_columns:

        return {
            "rows": len(data),
            "features": 0,
            "missing_values": 0,
        }


    feature_data = data[
        feature_columns
    ]


    return {

        "rows":
            len(data),

        "features":
            len(feature_columns),

        "missing_values":
            int(
                feature_data
                .isna()
                .sum()
                .sum()
            ),

        "start_date":
            str(
                data.index.min().date()
            ),

        "end_date":
            str(
                data.index.max().date()
            ),
    }


# ============================================================
# 市場特徴量の状態確認
# ============================================================

def get_available_market_features(
    data
):
    """
    どの市場特徴量が実際に利用可能か確認する。
    """

    if data is None or data.empty:

        return {}


    result = {}


    for market_name in MARKET_NAMES:

        prefix = (
            market_name + "_"
        )


        columns = [
            column
            for column in data.columns
            if column.startswith(
                prefix
            )
        ]


        result[
            market_name
        ] = {

            "available":
                len(columns) > 0,

            "feature_count":
                len(columns),

            "columns":
                columns,
        }


    return result
