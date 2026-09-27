# ============================================================
# アドバンテスト AI売買システム
# data/stock_data.py
#
# 株価データを取得するためのプログラム
#
# このファイルの役割
# ・Yahoo! Financeから株価データを取得
# ・データを整理
# ・異常なデータを除外
# ・他のプログラムから利用できる形にする
#
# ※売買判断はここでは行いません
# ============================================================


import yfinance as yf
import pandas as pd

from config import STOCK_CODE, DATA_PERIOD


# ============================================================
# 株価データ取得
# ============================================================

def get_stock_data(
    stock_code=STOCK_CODE,
    period=DATA_PERIOD,
    interval="1d"
):
    """
    株価データを取得する

    Parameters
    ----------
    stock_code : str
        Yahoo! Financeの銘柄コード

    period : str
        取得期間
        例：
        1y  = 1年
        2y  = 2年
        5y  = 5年
        10y = 10年

    interval : str
        データ間隔
        1d = 日足

    Returns
    -------
    pandas.DataFrame
        株価データ
    """

    try:

        # ----------------------------------------------------
        # Yahoo! Financeからデータ取得
        # ----------------------------------------------------

        data = yf.download(
            stock_code,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            multi_level_index=False
        )

        # ----------------------------------------------------
        # データが取得できなかった場合
        # ----------------------------------------------------

        if data is None or data.empty:

            raise ValueError(
                f"{stock_code} の株価データを取得できませんでした。"
            )

        # ----------------------------------------------------
        # 必要な列
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        # ----------------------------------------------------
        # 必要列が存在するか確認
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            raise ValueError(
                "必要な株価データがありません："
                + ", ".join(missing_columns)
            )

        # ----------------------------------------------------
        # 必要な列だけ使用
        # ----------------------------------------------------

        data = data[required_columns].copy()

        # ----------------------------------------------------
        # 数値データに変換
        # ----------------------------------------------------

        for column in required_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # 欠損データを削除
        # ----------------------------------------------------

        data.dropna(
            subset=required_columns,
            inplace=True
        )

        # ----------------------------------------------------
        # 日付順に並べる
        # ----------------------------------------------------

        data.sort_index(
            inplace=True
        )

        # ----------------------------------------------------
        # データ件数確認
        # ----------------------------------------------------

        if len(data) == 0:

            raise ValueError(
                "有効な株価データがありません。"
            )

        return data

    except Exception as e:

        # エラー内容を上位プログラムへ渡す
        raise RuntimeError(
            f"株価データ取得中にエラーが発生しました：{e}"
        ) from e


# ============================================================
# 最新株価取得
# ============================================================

def get_latest_price(
    stock_code=STOCK_CODE
):
    """
    最新の終値を取得する
    """

    data = get_stock_data(
        stock_code=stock_code,
        period="5d",
        interval="1d"
    )

    if data.empty:

        return None

    latest_close = data["Close"].iloc[-1]

    return float(latest_close)


# ============================================================
# 最新データ取得
# ============================================================

def get_latest_data(
    stock_code=STOCK_CODE
):
    """
    最新の株価データ1行を取得する
    """

    data = get_stock_data(
        stock_code=stock_code,
        period="5d",
        interval="1d"
    )

    if data.empty:

        return None

    return data.iloc[-1]


# ============================================================
# データ情報取得
# ============================================================

def get_data_info(
    data
):
    """
    取得したデータの基本情報を返す
    """

    if data is None or data.empty:

        return {
            "rows": 0,
            "start_date": None,
            "end_date": None
        }

    return {
        "rows": len(data),
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
