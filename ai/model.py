# ============================================================
# アドバンテスト AI売買システム
# ai/model.py
#
# 市場データ対応 AI予測モデル
#
# 使用する情報
# ・アドバンテストのテクニカル指標
# ・日経平均
# ・NASDAQ
# ・SOXX
# ・ドル円
#
# 目的
# ・翌営業日に終値が上昇する確率を予測
#
# モデル
# ・RandomForestClassifier
#
# 重要
# ・時系列順で学習 / テストを分割
# ・ランダムシャッフルしない
# ・未来の終値は正解ラベル作成にのみ使用
# ・実際の売買注文は行わない
# ============================================================


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# features.py
# ============================================================

from ai.features import (
    get_ai_feature_columns
)


# ============================================================
# AIモデル
# ============================================================

class StockPredictionModel:

    """
    アドバンテスト翌営業日上昇確率予測モデル
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42
    ):

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        self.model = RandomForestClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_split=min_samples_split,

            min_samples_leaf=min_samples_leaf,

            max_features=max_features,

            class_weight="balanced",

            random_state=random_state,

            n_jobs=-1
        )


        # ----------------------------------------------------
        # 学習に使用した特徴量
        # ----------------------------------------------------

        self.feature_columns = []


        # ----------------------------------------------------
        # 学習済みか
        # ----------------------------------------------------

        self.is_trained = False


        # ----------------------------------------------------
        # 評価指標
        # ----------------------------------------------------

        self.metrics = {}


        # ----------------------------------------------------
        # テスト結果
        # ----------------------------------------------------

        self.test_results = None


    # ========================================================
    # 正解ラベル作成
    # ========================================================

    @staticmethod
    def create_target(
        data
    ):

        """
        翌営業日の終値が
        当日の終値より高い場合

        1 = 上昇
        0 = 下落または同値

        最終日は翌営業日の終値が存在しないため
        NaNにする。
        """

        if data is None or data.empty:

            raise ValueError(
                "正解ラベルを作成するデータがありません。"
            )


        if "Close" not in data.columns:

            raise ValueError(
                "Close列がありません。"
            )


        # ----------------------------------------------------
        # 翌営業日終値
        # ----------------------------------------------------

        next_close = (
            data["Close"]
            .shift(-1)
        )


        # ----------------------------------------------------
        # 上昇 / 下落
        # ----------------------------------------------------

        target = (
            next_close
            > data["Close"]
        ).astype(float)


        # ----------------------------------------------------
        # 最終日は正解不明
        # ----------------------------------------------------

        target.loc[
            next_close.isna()
        ] = np.nan


        target.name = "Target"


        return target


    # ========================================================
    # 学習データ準備
    # ========================================================

    def prepare_training_data(
        self,
        data,
        feature_columns=None
    ):

        """
        AI学習用

        X = 特徴量
        y = 翌営業日上昇 / 下落

        を作成する。
        """

        if data is None or data.empty:

            raise ValueError(
                "AI学習用データがありません。"
            )


        df = data.copy()


        # ----------------------------------------------------
        # 日付順に並べる
        # ----------------------------------------------------

        df.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # AI特徴量取得
        # ----------------------------------------------------

        if feature_columns is None:

            feature_columns = (
                get_ai_feature_columns(
                    df
                )
            )


        if not feature_columns:

            raise ValueError(
                "AIで使用できる特徴量がありません。"
            )


        # ----------------------------------------------------
        # 重複削除
        # ----------------------------------------------------

        feature_columns = list(
            dict.fromkeys(
                feature_columns
            )
        )


        # ----------------------------------------------------
        # 存在確認
        # ----------------------------------------------------

        missing_columns = [

            column

            for column
            in feature_columns

            if column
            not in df.columns
        ]


        if missing_columns:

            raise ValueError(

                "AIに必要な特徴量がありません："

                + ", ".join(
                    missing_columns
                )
            )


        # ----------------------------------------------------
        # 特徴量
        # ----------------------------------------------------

        X = df[
            feature_columns
        ].copy()


        # ----------------------------------------------------
        # 数値へ変換
        # ----------------------------------------------------

        for column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # 無限大をNaN
        # ----------------------------------------------------

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )


        # ----------------------------------------------------
        # 正解ラベル
        # ----------------------------------------------------

        y = self.create_target(
            df
        )


        # ----------------------------------------------------
        # 特徴量と正解が揃っている日のみ使用
        # ----------------------------------------------------

        valid_mask = (
            X.notna().all(axis=1)
            & y.notna()
        )


        X = X.loc[
            valid_mask
        ].copy()


        y = y.loc[
            valid_mask
        ].copy()


        # ----------------------------------------------------
        # データ件数確認
        # ----------------------------------------------------

        if len(X) < 100:

            raise ValueError(

                "AI学習に使用できるデータが"

                f"{len(X)}件しかありません。"

                "100件以上必要です。"
            )


        # ----------------------------------------------------
        # 型
        # ----------------------------------------------------

        X = X.astype(float)

        y = y.astype(int)


        # ----------------------------------------------------
        # 特徴量保存
        # ----------------------------------------------------

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
        古いデータ → AI学習

        新しいデータ → AIテスト

        ランダムシャッフルはしない。
        """

        if not 0.5 <= train_ratio < 1.0:

            raise ValueError(
                "train_ratioは0.5以上1.0未満にしてください。"
            )


        split_index = int(
            len(X)
            * train_ratio
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
        Random Forestを学習し、
        新しい20%程度のデータで検証する。
        """

        # ----------------------------------------------------
        # 学習データ作成
        # ----------------------------------------------------

        X, y = (
            self.prepare_training_data(

                data=data,

                feature_columns=feature_columns
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

            X=X,

            y=y,

            train_ratio=train_ratio
        )


        # ----------------------------------------------------
        # 学習データに両クラスがあるか確認
        # ----------------------------------------------------

        if y_train.nunique() < 2:

            raise ValueError(
                "学習データに上昇・下落の両方がありません。"
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


        # ----------------------------------------------------
        # 上昇確率
        # ----------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                X_test
            )[:, 1]
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = (
            precision_score(

                y_test,

                predictions,

                zero_division=0
            )
        )


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = (
            recall_score(

                y_test,

                predictions,

                zero_division=0
            )
        )


        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1 = (
            f1_score(

                y_test,

                predictions,

                zero_division=0
            )
        )


        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        try:

            if y_test.nunique() >= 2:

                auc = (
                    roc_auc_score(

                        y_test,

                        probabilities
                    )
                )

            else:

                auc = np.nan


        except ValueError:

            auc = np.nan


        # ----------------------------------------------------
        # 混同行列
        # ----------------------------------------------------

        cm = confusion_matrix(

            y_test,

            predictions,

            labels=[
                0,
                1
            ]
        )


        true_negative = int(
            cm[0, 0]
        )

        false_positive = int(
            cm[0, 1]
        )

        false_negative = int(
            cm[1, 0]
        )

        true_positive = int(
            cm[1, 1]
        )


        # ----------------------------------------------------
        # 上昇率
        # ----------------------------------------------------

        train_up_rate = float(
            y_train.mean()
        )

        test_up_rate = float(
            y_test.mean()
        )


        # ----------------------------------------------------
        # 評価結果
        # ----------------------------------------------------

        self.metrics = {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "auc":
                (
                    float(auc)

                    if not np.isnan(auc)

                    else None
                ),

            "train_samples":
                int(
                    len(X_train)
                ),

            "test_samples":
                int(
                    len(X_test)
                ),

            "feature_count":
                int(
                    len(
                        self.feature_columns
                    )
                ),

            "train_up_rate":
                train_up_rate,

            "test_up_rate":
                test_up_rate,

            "true_negative":
                true_negative,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "true_positive":
                true_positive,
        }


        # ----------------------------------------------------
        # テスト結果保存
        # ----------------------------------------------------

        self.test_results = pd.DataFrame(

            {

                "Actual":
                    y_test,

                "Prediction":
                    predictions,

                "Probability_Up":
                    probabilities
            },

            index=X_test.index
        )


        return self.metrics


    # ========================================================
    # 最新データ作成
    # ========================================================

    def prepare_latest_features(
        self,
        data
    ):

        """
        最新日の特徴量をAI予測用に準備する。
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
        # 必要な特徴量確認
        # ----------------------------------------------------

        missing_columns = [

            column

            for column
            in self.feature_columns

            if column
            not in data.columns
        ]


        if missing_columns:

            raise ValueError(

                "最新予測に必要な特徴量がありません："

                + ", ".join(
                    missing_columns
                )
            )


        # ----------------------------------------------------
        # 最新行
        # ----------------------------------------------------

        latest = data.iloc[
            [-1]
        ].copy()


        X_latest = latest[
            self.feature_columns
        ].copy()


        # ----------------------------------------------------
        # 数値化
        # ----------------------------------------------------

        for column in X_latest.columns:

            X_latest[column] = (
                pd.to_numeric(

                    X_latest[column],

                    errors="coerce"
                )
            )


        # ----------------------------------------------------
        # 無限大
        # ----------------------------------------------------

        X_latest.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True
        )


        # ----------------------------------------------------
        # 欠損値確認
        # ----------------------------------------------------

        missing_latest = [

            column

            for column
            in X_latest.columns

            if X_latest[column]
            .isna()
            .any()
        ]


        if missing_latest:

            raise ValueError(

                "最新日のAI特徴量に欠損があります："

                + ", ".join(
                    missing_latest
                )
            )


        return X_latest


    # ========================================================
    # 上昇確率
    # ========================================================

    def predict_probability(
        self,
        data
    ):

        """
        最新日の特徴量から
        翌営業日の上昇確率を予測する。
        """

        X_latest = (
            self.prepare_latest_features(
                data
            )
        )


        probabilities = (
            self.model.predict_proba(
                X_latest
            )
        )


        # ----------------------------------------------------
        # class 1 の位置を安全に取得
        # ----------------------------------------------------

        classes = list(
            self.model.classes_
        )


        if 1 not in classes:

            raise RuntimeError(
                "AIモデルに上昇クラスがありません。"
            )


        up_index = (
            classes.index(1)
        )


        probability_up = float(
            probabilities[
                0,
                up_index
            ]
        )


        return probability_up


    # ========================================================
    # 最新予測
    # ========================================================

    def predict(
        self,
        data,
        threshold=0.5
    ):

        """
        最新日の

        ・上昇確率
        ・下落確率
        ・方向予測

        を返す。
        """

        probability_up = (
            self.predict_probability(
                data
            )
        )


        probability_down = (
            1.0
            - probability_up
        )


        prediction = (

            1

            if probability_up
            >= threshold

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
                (
                    "上昇"

                    if prediction == 1

                    else "下落"
                ),

            "threshold":
                float(
                    threshold
                ),
        }


    # ========================================================
    # 特徴量重要度
    # ========================================================

    def get_feature_importance(
        self
    ):

        """
        Random Forestが
        どの特徴量をどれだけ使用したかを返す。
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
    # テスト結果
    # ========================================================

    def get_test_results(
        self
    ):

        """
        AIのテスト期間における
        日別予測結果を返す。
        """

        if not self.is_trained:

            raise RuntimeError(
                "AIモデルがまだ学習されていません。"
            )


        if self.test_results is None:

            return pd.DataFrame()


        return (
            self.test_results.copy()
        )


    # ========================================================
    # モデル情報
    # ========================================================

    def get_model_info(
        self
    ):

        """
        AIモデルの情報を返す。
        """

        return {

            "model":
                "RandomForestClassifier",

            "trained":
                self.is_trained,

            "feature_count":
                len(
                    self.feature_columns
                ),

            "feature_columns":
                self.feature_columns.copy(),

            "metrics":
                self.metrics.copy(),
        }


# ============================================================
# AIを簡単に学習する関数
# ============================================================

def train_stock_model(
    data,
    feature_columns=None,
    train_ratio=0.8
):

    """
    AIモデルを作成し、
    学習まで一度に行う。
    """

    model = (
        StockPredictionModel()
    )


    metrics = model.train(

        data=data,

        feature_columns=feature_columns,

        train_ratio=train_ratio
    )


    return (
        model,
        metrics
    )


# ============================================================
# 最新AI予測
# ============================================================

def predict_stock_probability(
    model,
    data,
    threshold=0.5
):

    """
    学習済みAIモデルから
    最新予測を取得する。
    """

    return model.predict(

        data=data,

        threshold=threshold
    )
