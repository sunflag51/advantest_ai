# ============================================================
# アドバンテスト AI売買システム
# backtest/engine.py
#
# ウォークフォワード検証エンジン
#
# 目的
# ・未来データを使わずAIを検証する
# ・過去データだけで学習
# ・次の期間を予測
# ・時間を進めて再学習
#
# 現段階では
# 「AI予測能力の検証」が目的
#
# 実際の売買注文は行わない
# ============================================================


import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from ai.model import StockPredictionModel

from ai.features import (
    get_ai_feature_columns
)


# ============================================================
# ウォークフォワード検証エンジン
# ============================================================

class WalkForwardBacktest:

    """
    ウォークフォワード方式で
    AIの予測性能を検証するクラス
    """

    # ========================================================
    # 初期化
    # ========================================================

    def __init__(
        self,
        initial_train_size=500,
        test_size=20,
        retrain_every=20,
        threshold=0.50
    ):

        """
        initial_train_size
            最初のAI学習に使用する営業日数

        test_size
            1回の学習後に検証する最大営業日数

        retrain_every
            何営業日ごとにAIを再学習するか

        threshold
            上昇と判定する確率
        """

        self.initial_train_size = int(
            initial_train_size
        )

        self.test_size = int(
            test_size
        )

        self.retrain_every = int(
            retrain_every
        )

        self.threshold = float(
            threshold
        )


        if self.initial_train_size < 100:

            raise ValueError(
                "initial_train_sizeは"
                "100以上にしてください。"
            )


        if self.test_size < 1:

            raise ValueError(
                "test_sizeは1以上にしてください。"
            )


        if self.retrain_every < 1:

            raise ValueError(
                "retrain_everyは1以上にしてください。"
            )


        if not 0.0 < self.threshold < 1.0:

            raise ValueError(
                "thresholdは0〜1の間にしてください。"
            )


        # ----------------------------------------------------
        # 結果保存
        # ----------------------------------------------------

        self.results = pd.DataFrame()

        self.metrics = {}

        self.feature_columns = []

        self.training_log = []


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        data,
        feature_columns=None
    ):

        """
        ウォークフォワード検証用データを準備する。

        Target
        1 = 翌営業日上昇
        0 = 翌営業日下落または同値
        """

        if data is None or data.empty:

            raise ValueError(
                "バックテスト用データがありません。"
            )


        df = data.copy()


        # ----------------------------------------------------
        # 時系列順
        # ----------------------------------------------------

        df.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # 特徴量
        # ----------------------------------------------------

        if feature_columns is None:

            feature_columns = (
                get_ai_feature_columns(
                    df
                )
            )


        if not feature_columns:

            raise ValueError(
                "AI特徴量がありません。"
            )


        feature_columns = list(
            dict.fromkeys(
                feature_columns
            )
        )


        missing_columns = [

            column

            for column in feature_columns

            if column not in df.columns
        ]


        if missing_columns:

            raise ValueError(

                "必要な特徴量がありません："

                + ", ".join(
                    missing_columns
                )
            )


        if "Close" not in df.columns:

            raise ValueError(
                "Close列がありません。"
            )


        # ----------------------------------------------------
        # 特徴量を数値化
        # ----------------------------------------------------

        for column in feature_columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
            )


        # ----------------------------------------------------
        # Close数値化
        # ----------------------------------------------------

        df["Close"] = (
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            )
        )


        # ----------------------------------------------------
        # 無限大処理
        # ----------------------------------------------------

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )


        # ----------------------------------------------------
        # 翌営業日終値
        # ----------------------------------------------------

        df["Next_Close"] = (
            df["Close"]
            .shift(-1)
        )


        # ----------------------------------------------------
        # 翌営業日リターン
        # ----------------------------------------------------

        df["Next_Return"] = (
            df["Next_Close"]
            / df["Close"]
            - 1
        )


        # ----------------------------------------------------
        # 正解
        # ----------------------------------------------------

        df["Target"] = np.where(

            df["Next_Close"].notna(),

            (
                df["Next_Close"]
                > df["Close"]
            ).astype(int),

            np.nan
        )


        # ----------------------------------------------------
        # 必要列が揃った日のみ
        # ----------------------------------------------------

        required_columns = (
            feature_columns
            + [
                "Close",
                "Next_Close",
                "Next_Return",
                "Target"
            ]
        )


        df = df.dropna(
            subset=required_columns
        ).copy()


        if len(df) <= self.initial_train_size:

            raise ValueError(

                "ウォークフォワード検証に必要な"
                "データが不足しています。"

                f" 使用可能データ={len(df)}件、"

                f" initial_train_size="
                f"{self.initial_train_size}件"
            )


        df["Target"] = (
            df["Target"]
            .astype(int)
        )


        self.feature_columns = (
            feature_columns.copy()
        )


        return df


    # ========================================================
    # AIモデル作成
    # ========================================================

    @staticmethod
    def create_model():

        """
        各ウォークフォワード学習で
        新しいAIモデルを作成する。
        """

        return StockPredictionModel()


    # ========================================================
    # AIを過去データだけで学習
    # ========================================================

    def train_model(
        self,
        train_data
    ):

        """
        train_dataのみを使って
        Random Forestを学習する。

        StockPredictionModel.train()は内部で
        さらに80/20分割するため、
        ウォークフォワードでは直接
        sklearnモデルを学習させる。
        """

        model = self.create_model()


        X_train = train_data[
            self.feature_columns
        ].copy()


        y_train = train_data[
            "Target"
        ].copy()


        # ----------------------------------------------------
        # 上昇 / 下落の両クラス確認
        # ----------------------------------------------------

        if y_train.nunique() < 2:

            raise ValueError(
                "学習期間に上昇・下落の"
                "両クラスがありません。"
            )


        # ----------------------------------------------------
        # 学習
        # ----------------------------------------------------

        model.feature_columns = (
            self.feature_columns.copy()
        )


        model.model.fit(
            X_train,
            y_train
        )


        model.is_trained = True


        return model


    # ========================================================
    # ウォークフォワード実行
    # ========================================================

    def run(
        self,
        data,
        feature_columns=None
    ):

        """
        ウォークフォワード検証を実行する。

        例

        過去500日
            ↓
        AI学習
            ↓
        次の20日を予測
            ↓
        20日進む
            ↓
        過去520日で再学習
            ↓
        次の20日を予測
            ↓
        繰り返し
        """

        # ----------------------------------------------------
        # データ準備
        # ----------------------------------------------------

        df = self.prepare_data(
            data=data,
            feature_columns=feature_columns
        )


        result_rows = []

        self.training_log = []


        # ----------------------------------------------------
        # 最初のテスト位置
        # ----------------------------------------------------

        test_start = (
            self.initial_train_size
        )


        model_number = 0


        # ====================================================
        # 時間を前へ進める
        # ====================================================

        while test_start < len(df):

            # ------------------------------------------------
            # 学習期間
            #
            # expanding window方式
            # 過去データをすべて使用
            # ------------------------------------------------

            train_data = df.iloc[
                :test_start
            ].copy()


            # ------------------------------------------------
            # テスト期間終了位置
            # ------------------------------------------------

            test_end = min(

                test_start
                + self.test_size,

                len(df)
            )


            # ------------------------------------------------
            # テスト期間
            # ------------------------------------------------

            test_data = df.iloc[
                test_start:test_end
            ].copy()


            if test_data.empty:

                break


            # ------------------------------------------------
            # AI学習
            # ------------------------------------------------

            model = self.train_model(
                train_data
            )


            model_number += 1


            # ------------------------------------------------
            # 学習ログ
            # ------------------------------------------------

            self.training_log.append(

                {

                    "model_number":
                        model_number,

                    "train_start":
                        train_data.index[0],

                    "train_end":
                        train_data.index[-1],

                    "train_samples":
                        len(train_data),

                    "test_start":
                        test_data.index[0],

                    "test_end":
                        test_data.index[-1],

                    "test_samples":
                        len(test_data)
                }
            )


            # =================================================
            # テスト期間を1日ずつ予測
            # =================================================

            for date, row in test_data.iterrows():

                X_test = pd.DataFrame(

                    [
                        row[
                            self.feature_columns
                        ].values
                    ],

                    columns=self.feature_columns,

                    index=[date]
                )


                X_test = X_test.astype(
                    float
                )


                # --------------------------------------------
                # 上昇確率
                # --------------------------------------------

                probability_matrix = (
                    model.model.predict_proba(
                        X_test
                    )
                )


                classes = list(
                    model.model.classes_
                )


                if 1 not in classes:

                    continue


                up_index = (
                    classes.index(1)
                )


                probability_up = float(

                    probability_matrix[
                        0,
                        up_index
                    ]
                )


                probability_down = (
                    1.0
                    - probability_up
                )


                # --------------------------------------------
                # AI方向予測
                # --------------------------------------------

                prediction = (

                    1

                    if probability_up
                    >= self.threshold

                    else 0
                )


                # --------------------------------------------
                # 結果保存
                # --------------------------------------------

                result_rows.append(

                    {

                        "Date":
                            date,

                        "Model_Number":
                            model_number,

                        "Close":
                            float(
                                row["Close"]
                            ),

                        "Next_Close":
                            float(
                                row["Next_Close"]
                            ),

                        "Next_Return":
                            float(
                                row["Next_Return"]
                            ),

                        "Actual":
                            int(
                                row["Target"]
                            ),

                        "Prediction":
                            int(
                                prediction
                            ),

                        "Probability_Up":
                            probability_up,

                        "Probability_Down":
                            probability_down,

                        "Correct":
                            int(
                                prediction
                                ==
                                int(
                                    row["Target"]
                                )
                            )
                    }
                )


            # ------------------------------------------------
            # 次の期間へ
            #
            # 現在はtest_sizeとretrain_everyの
            # 小さい方を使って進める。
            # ------------------------------------------------

            step_size = min(
                self.test_size,
                self.retrain_every
            )


            test_start += (
                step_size
            )


        # ====================================================
        # 結果DataFrame
        # ====================================================

        self.results = pd.DataFrame(
            result_rows
        )


        if self.results.empty:

            raise ValueError(
                "ウォークフォワード結果がありません。"
            )


        self.results.set_index(
            "Date",
            inplace=True
        )


        self.results.sort_index(
            inplace=True
        )


        # ----------------------------------------------------
        # 同じ日が重複した場合
        #
        # test_sizeとretrain_everyが異なる場合の
        # 重複を安全に処理
        # ----------------------------------------------------

        self.results = self.results[
            ~self.results.index.duplicated(
                keep="last"
            )
        ]


        # ----------------------------------------------------
        # 評価指標
        # ----------------------------------------------------

        self.metrics = (
            self.calculate_metrics(
                self.results
            )
        )


        return (
            self.results.copy(),
            self.metrics.copy()
        )


    # ========================================================
    # 評価指標
    # ========================================================

    def calculate_metrics(
        self,
        results
    ):

        """
        ウォークフォワード全期間の
        AI予測能力を評価する。
        """

        if results is None or results.empty:

            return {}


        y_true = (
            results["Actual"]
            .astype(int)
        )


        y_pred = (
            results["Prediction"]
            .astype(int)
        )


        probabilities = (
            results["Probability_Up"]
            .astype(float)
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = (
            accuracy_score(
                y_true,
                y_pred
            )
        )


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = (
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = (
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1 = (
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        try:

            if y_true.nunique() >= 2:

                auc = (
                    roc_auc_score(
                        y_true,
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
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        )


        tn = int(
            cm[0, 0]
        )

        fp = int(
            cm[0, 1]
        )

        fn = int(
            cm[1, 0]
        )

        tp = int(
            cm[1, 1]
        )


        # ----------------------------------------------------
        # 実際の上昇日割合
        # ----------------------------------------------------

        actual_up_rate = float(
            y_true.mean()
        )


        # ----------------------------------------------------
        # AIが上昇と予測した割合
        # ----------------------------------------------------

        predicted_up_rate = float(
            y_pred.mean()
        )


        # ----------------------------------------------------
        # 上昇予測日の平均翌日リターン
        # ----------------------------------------------------

        predicted_up_returns = (
            results.loc[
                results["Prediction"] == 1,
                "Next_Return"
            ]
        )


        if len(predicted_up_returns) > 0:

            average_return_when_up = float(
                predicted_up_returns.mean()
            )

        else:

            average_return_when_up = 0.0


        # ----------------------------------------------------
        # 下落予測日の平均翌日リターン
        # ----------------------------------------------------

        predicted_down_returns = (
            results.loc[
                results["Prediction"] == 0,
                "Next_Return"
            ]
        )


        if len(predicted_down_returns) > 0:

            average_return_when_down = float(
                predicted_down_returns.mean()
            )

        else:

            average_return_when_down = 0.0


        # ----------------------------------------------------
        # 高確率予測
        #
        # 60%以上
        # ----------------------------------------------------

        high_confidence = results[
            results["Probability_Up"]
            >= 0.60
        ]


        if len(high_confidence) > 0:

            high_confidence_accuracy = float(
                high_confidence[
                    "Correct"
                ].mean()
            )

            high_confidence_return = float(
                high_confidence[
                    "Next_Return"
                ].mean()
            )

        else:

            high_confidence_accuracy = None

            high_confidence_return = None


        # ----------------------------------------------------
        # 評価結果
        # ----------------------------------------------------

        metrics = {

            "samples":
                int(
                    len(results)
                ),

            "accuracy":
                float(
                    accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "auc":
                (
                    float(auc)

                    if not np.isnan(auc)

                    else None
                ),

            "actual_up_rate":
                actual_up_rate,

            "predicted_up_rate":
                predicted_up_rate,

            "average_return_when_up":
                average_return_when_up,

            "average_return_when_down":
                average_return_when_down,

            "high_confidence_samples":
                int(
                    len(high_confidence)
                ),

            "high_confidence_accuracy":
                high_confidence_accuracy,

            "high_confidence_return":
                high_confidence_return,

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            "model_count":
                int(
                    len(
                        self.training_log
                    )
                ),

            "threshold":
                float(
                    self.threshold
                )
        }


        return metrics


    # ========================================================
    # 学習ログ
    # ========================================================

    def get_training_log(
        self
    ):

        """
        AIがいつ、どの期間で
        再学習したかを取得する。
        """

        if not self.training_log:

            return pd.DataFrame()


        return pd.DataFrame(
            self.training_log
        )


    # ========================================================
    # 結果取得
    # ========================================================

    def get_results(
        self
    ):

        """
        ウォークフォワード結果を取得する。
        """

        return (
            self.results.copy()
        )


    # ========================================================
    # 高確率予測のみ取得
    # ========================================================

    def get_high_confidence_results(
        self,
        probability=0.60
    ):

        """
        指定した上昇確率以上の
        AI予測だけ取得する。
        """

        if self.results.empty:

            return pd.DataFrame()


        return (
            self.results[
                self.results[
                    "Probability_Up"
                ]
                >= probability
            ]
            .copy()
        )


# ============================================================
# 簡単実行関数
# ============================================================

def run_walk_forward_backtest(
    data,
    feature_columns=None,
    initial_train_size=500,
    test_size=20,
    retrain_every=20,
    threshold=0.50
):

    """
    ウォークフォワード検証を
    一度に実行する便利関数。
    """

    engine = WalkForwardBacktest(

        initial_train_size=
            initial_train_size,

        test_size=
            test_size,

        retrain_every=
            retrain_every,

        threshold=
            threshold
    )


    results, metrics = (
        engine.run(

            data=data,

            feature_columns=
                feature_columns
        )
    )


    return (
        engine,
        results,
        metrics
    )
