# ============================================================
# アドバンテスト AI売買システム
# ai/model.py
#
# AI予測モデル
#
# 目的
# ・過去の株価データからAIを学習
# ・翌営業日に株価が上昇する確率を予測
#
# 今回使用するモデル
# Random Forest
#
# 重要
# ・このファイルでは実際の売買注文を行いません
# ・「買い」「売り」を直接決定しません
# ・AIは上昇確率を予測します
# ============================================================


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# 使用する特徴量
# ============================================================

DEFAULT_FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # 価格変化率
    # --------------------------------------------------------

    "Return_1D",
    "Return_5D",
    "Return_20D",
    "Return_60D",

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    "RSI_14",

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # --------------------------------------------------------
    # ボリンジャーバンド
    # --------------------------------------------------------

    "BB_Width",
    "BB_Position",

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    "ATR_14",

    # --------------------------------------------------------
    # 出来高
    # --------------------------------------------------------

    "Volume_Ratio",

    # --------------------------------------------------------
    # ボラティリティ
    # --------------------------------------------------------

    "Volatility_20D",

    # --------------------------------------------------------
    # トレンド
    # --------------------------------------------------------

    "Trend_5_25",
    "Trend_25_75",

    "Price_vs_SMA25",
    "Price_vs_SMA75"
]


# ============================================================
# AIモデルクラス
# ============================================================

class StockPredictionModel:
    """
    株価上昇確率予測モデル
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ):
        """
        Random Forestモデルを作成
        """

        self.model = RandomForestClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_leaf=min_samples_leaf,

            random_state=random_state,

            class_weight="balanced",

            n_jobs=-1
        )

        self.feature_columns = []

        self.is_trained = False

        self.metrics = {}


    # ========================================================
    # 正解データ作成
    # ========================================================

    @staticmethod
    def create_target(
        data
    ):
        """
        翌営業日の終値が現在の終値より高い場合

        1 = 上昇
        0 = 下落または同値

        とする。

        重要：
        現在の日の情報だけを使って
        翌営業日の結果を予測する。
        """

        df = data.copy()

        next_close = (
            df["Close"]
            .shift(-1)
        )

        target = (
            next_close >
            df["Close"]
        ).astype(int)

        # 最終日は翌日の価格が存在しないためNaN扱い
        target.iloc[-1] = np.nan

        return target


    # ========================================================
    # 学習データ作成
    # ========================================================

    def prepare_training_data(
        self,
        data,
        feature_columns=None
    ):
        """
        AI学習用データを作成する
        """

        if data is None or data.empty:

            raise ValueError(
                "学習用データが空です。"
            )


        df = data.copy()


        # ----------------------------------------------------
        # 使用する特徴量
        # ----------------------------------------------------

        if feature_columns is None:

            feature_columns = (
                DEFAULT_FEATURE_COLUMNS.copy()
            )


        # ----------------------------------------------------
        # 実際に存在する列だけ確認
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in feature_columns
            if column not in df.columns
        ]


        if missing_columns:

            raise ValueError(
                "AIに必要な特徴量がありません："
                + ", ".join(missing_columns)
            )


        # ----------------------------------------------------
        # 特徴量
        # ----------------------------------------------------

        X = df[
            feature_columns
        ].copy()


        # ----------------------------------------------------
        # 正解データ
        # ----------------------------------------------------

        y = self.create_target(
            df
        )


        # ----------------------------------------------------
        # 無限大をNaNにする
        # ----------------------------------------------------

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )


        # ----------------------------------------------------
        # 欠損値を削除
        #
        # RSIや移動平均などは
        # 最初の数日間にNaNが発生する
        # ----------------------------------------------------

        valid_mask = (
            X.notna().all(axis=1)
            & y.notna()
        )


        X = X.loc[
            valid_mask
        ]

        y = y.loc[
            valid_mask
        ]


        if len(X) < 100:

            raise ValueError(
                "AI学習に必要なデータが不足しています。"
                "最低100件以上を推奨します。"
            )


        # ----------------------------------------------------
        # 型を整理
        # ----------------------------------------------------

        X = X.astype(float)

        y = y.astype(int)


        self.feature_columns = (
            feature_columns.copy()
        )


        return X, y


    # ========================================================
    # 時系列分割
    # ========================================================

    @staticmethod
    def time_series_split(
        X,
        y,
        train_ratio=0.8
    ):
        """
        時系列データを

        過去80% → 学習
        新しい20% → テスト

        に分割する。

        通常のランダム分割ではなく、
        時系列順を維持する。
        """

        split_index = int(
            len(X) * train_ratio
        )


        if split_index <= 0:

            raise ValueError(
                "学習データがありません。"
            )


        if split_index >= len(X):

            raise ValueError(
                "テストデータがありません。"
            )


        X_train = X.iloc[
            :split_index
        ].copy()


        X_test = X.iloc[
            split_index:
        ].copy()


        y_train = y.iloc[
            :split_index
        ].copy()


        y_test = y.iloc[
            split_index:
        ].copy()


        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


    # ========================================================
    # AI学習
    # ========================================================

    def train(
        self,
        data,
        feature_columns=None,
        train_ratio=0.8
    ):
        """
        AIモデルを学習する
        """

        # ----------------------------------------------------
        # 学習データ作成
        # ----------------------------------------------------

        X, y = (
            self.prepare_training_data(
                data,
                feature_columns
            )
        )


        # ----------------------------------------------------
        # 時系列分割
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.time_series_split(
            X,
            y,
            train_ratio
        )


        # ----------------------------------------------------
        # AI学習
        # ----------------------------------------------------

        self.model.fit(
            X_train,
            y_train
        )


        self.is_trained = True


        # ----------------------------------------------------
        # テスト予測
        # ----------------------------------------------------

        predictions = (
            self.model.predict(
                X_test
            )
        )


        probabilities = (
            self.model.predict_proba(
                X_test
            )[:, 1]
        )


        # ----------------------------------------------------
        # 精度
        # ----------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )


        precision = (
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        )


        recall = (
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        try:

            auc = (
                roc_auc_score(
                    y_test,
                    probabilities
                )
            )

        except ValueError:

            auc = np.nan


        # ----------------------------------------------------
        # 指標保存
        # ----------------------------------------------------

        self.metrics = {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "auc":
                float(auc)
                if not np.isnan(auc)
                else None,

            "train_samples":
                len(X_train),

            "test_samples":
                len(X_test)
        }


        return self.metrics


    # ========================================================
    # 上昇確率予測
    # ========================================================

    def predict_probability(
        self,
        data
    ):
        """
        最新データから
        翌営業日の上昇確率を予測する
        """

        if not self.is_trained:

            raise RuntimeError(
                "AIモデルがまだ学習されていません。"
            )


        if data is None or data.empty:

            raise ValueError(
                "予測用データがありません。"
            )


        # ----------------------------------------------------
        # 最新行
        # ----------------------------------------------------

        latest = data.iloc[
            [-1]
        ].copy()


        # ----------------------------------------------------
        # 特徴量確認
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in latest.columns
        ]


        if missing_columns:

            raise ValueError(
                "予測に必要な特徴量がありません："
                + ", ".join(missing_columns)
            )


        X_latest = latest[
            self.feature_columns
        ].copy()


        # ----------------------------------------------------
        # 欠損値確認
        # ----------------------------------------------------

        if X_latest.isna().any().any():

            raise ValueError(
                "最新データに欠損値があります。"
            )


        # ----------------------------------------------------
        # 上昇確率
        # ----------------------------------------------------

        probability = (
            self.model.predict_proba(
                X_latest
            )[0, 1]
        )


        return float(
            probability
        )


    # ========================================================
    # 上昇・下落予測
    # ========================================================

    def predict(
        self,
        data
    ):
        """
        最新データから

        ・上昇確率
        ・下落確率
        ・予測

        を返す
        """

        if not self.is_trained:

            raise RuntimeError(
                "AIモデルがまだ学習されていません。"
            )


        probability_up = (
            self.predict_probability(
                data
            )
        )


        probability_down = (
            1 -
            probability_up
        )


        prediction = (
            1
            if probability_up >= 0.5
            else 0
        )


        return {

            "probability_up":
                probability_up,

            "probability_down":
                probability_down,

            "prediction":
                prediction,

            "prediction_text":
                "上昇"
                if prediction == 1
                else "下落"
        }


    # ========================================================
    # 特徴量重要度
    # ========================================================

    def get_feature_importance(
        self
    ):
        """
        AIがどの特徴量を重視したかを取得する
        """

        if not self.is_trained:

            raise RuntimeError(
                "AIモデルがまだ学習されていません。"
            )


        importance = (
            self.model.feature_importances_
        )


        result = pd.DataFrame(
            {
                "feature":
                    self.feature_columns,

                "importance":
                    importance
            }
        )


        result.sort_values(
            "importance",
            ascending=False,
            inplace=True
        )


        result.reset_index(
            drop=True,
            inplace=True
        )


        return result


    # ========================================================
    # モデル情報
    # ========================================================

    def get_model_info(
        self
    ):
        """
        モデル情報を取得する
        """

        return {

            "model":
                "RandomForestClassifier",

            "features":
                len(
                    self.feature_columns
                ),

            "trained":
                self.is_trained,

            "metrics":
                self.metrics
        }


# ============================================================
# 簡単にAIを学習する関数
# ============================================================

def train_stock_model(
    data,
    feature_columns=None,
    train_ratio=0.8
):
    """
    AIモデルを作成して学習する
    """

    model = StockPredictionModel()


    metrics = model.train(
        data=data,
        feature_columns=feature_columns,
        train_ratio=train_ratio
    )


    return model, metrics


# ============================================================
# 最新の上昇確率を取得
# ============================================================

def predict_stock_probability(
    model,
    data
):
    """
    学習済みモデルから
    最新の上昇確率を取得する
    """

    result = model.predict(
        data
    )


    return result
