# ============================================================
# backtest/engine.py
#
# WalkForwardBacktest v2
#
# 1日 / 3日 / 5日ターゲット対応
#
# AI/model.py v2 と同じTarget定義を使用
#
# Signal = t日の終値
# Entry  = Open(t+1)
# Target = Close(t+horizon)
#
# Future_Return
#   = Close(t+horizon) / Open(t+1) - 1
#
# Target = 1
#   Future_Return > target_return_threshold
#
# Target = 0
#   それ以外
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
    confusion_matrix,
)

from ai.features import get_ai_feature_columns


SUPPORTED_HORIZONS = (
    1,
    3,
    5,
)


class WalkForwardBacktest:

    def __init__(
        self,
        initial_train_size=500,
        test_size=20,
        retrain_every=20,
        threshold=0.50,

        target_horizon=3,
        target_return_threshold=0.0,

        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
    ):

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

        self.target_horizon = int(
            target_horizon
        )

        self.target_return_threshold = float(
            target_return_threshold
        )

        if (
            self.target_horizon
            not in SUPPORTED_HORIZONS
        ):

            raise ValueError(
                "target_horizon は "
                "1, 3, 5 のいずれかを"
                "指定してください。"
            )

        self.n_estimators = int(
            n_estimators
        )

        self.max_depth = int(
            max_depth
        )

        self.min_samples_split = int(
            min_samples_split
        )

        self.min_samples_leaf = int(
            min_samples_leaf
        )

        self.max_features = (
            max_features
        )

        self.random_state = int(
            random_state
        )

        self.results = pd.DataFrame()

        self.metrics = {}

        self.training_log = (
            pd.DataFrame()
        )

        self.feature_columns = []


    # ========================================================
    # モデル
    # ========================================================

    def create_model(
        self,
    ):

        return RandomForestClassifier(

            n_estimators=
                self.n_estimators,

            max_depth=
                self.max_depth,

            min_samples_split=
                self.min_samples_split,

            min_samples_leaf=
                self.min_samples_leaf,

            max_features=
                self.max_features,

            random_state=
                self.random_state,

            class_weight=
                "balanced",

            n_jobs=-1,
        )


    # ========================================================
    # データ準備
    # ========================================================

    def prepare_data(
        self,
        ai_data,
    ):

        if (
            ai_data is None
            or ai_data.empty
        ):

            raise ValueError(
                "AIデータがありません。"
            )

        if "Open" not in ai_data.columns:

            raise ValueError(
                "Open 列がありません。"
            )

        if "Close" not in ai_data.columns:

            raise ValueError(
                "Close 列がありません。"
            )


        data = ai_data.copy()

        data.index = pd.to_datetime(
            data.index
        )

        if getattr(
            data.index,
            "tz",
            None,
        ) is not None:

            data.index = (
                data.index
                .tz_localize(None)
            )

        data = data[
            ~data.index.duplicated(
                keep="last"
            )
        ].sort_index()


        # ====================================================
        # AI特徴量
        # ====================================================

        requested_features = (
            get_ai_feature_columns()
        )

        self.feature_columns = [

            column

            for column
            in requested_features

            if column in data.columns
        ]

        if not self.feature_columns:

            raise ValueError(
                "AI特徴量がありません。"
            )


        # ====================================================
        # 数値化
        # ====================================================

        for column in (
            self.feature_columns
            + [
                "Open",
                "Close",
            ]
        ):

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
            )


        # ====================================================
        # Target
        #
        # t日の終値でシグナル
        #
        # Entry:
        #   Open(t+1)
        #
        # Evaluation:
        #   Close(t+horizon)
        # ====================================================

        data[
            "Entry_Open"
        ] = data[
            "Open"
        ].shift(-1)


        data[
            "Future_Close"
        ] = data[
            "Close"
        ].shift(
            -self.target_horizon
        )


        data[
            "Future_Return"
        ] = (

            data[
                "Future_Close"
            ]

            /

            data[
                "Entry_Open"
            ]

            - 1.0
        )


        # ====================================================
        # Target
        # ====================================================

        valid_target = (

            data[
                "Entry_Open"
            ].notna()

            &

            data[
                "Future_Close"
            ].notna()

            &

            data[
                "Future_Return"
            ].notna()
        )


        data[
            "Target"
        ] = np.nan


        data.loc[
            valid_target,
            "Target",
        ] = (

            data.loc[
                valid_target,
                "Future_Return",
            ]

            >

            self.target_return_threshold

        ).astype(int)


        # ====================================================
        # 学習用
        # ====================================================

        required = (

            self.feature_columns

            + [
                "Entry_Open",
                "Future_Close",
                "Future_Return",
                "Target",
            ]
        )


        prepared = (

            data[
                required
            ]

            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

            .dropna()

            .copy()
        )


        prepared[
            "Target"
        ] = prepared[
            "Target"
        ].astype(int)


        if (
            len(prepared)
            <= self.initial_train_size
        ):

            raise ValueError(
                "ウォークフォワード検証に"
                "必要なデータ数が不足しています。"
            )


        return prepared


    # ========================================================
    # Positive Probability
    # ========================================================

    @staticmethod
    def positive_probability(
        model,
        X,
    ):

        probabilities = (
            model.predict_proba(
                X
            )
        )

        classes = list(
            model.classes_
        )

        if 1 not in classes:

            return np.zeros(
                len(X),
                dtype=float,
            )

        positive_index = (
            classes.index(1)
        )

        return probabilities[
            :,
            positive_index
        ]


    # ========================================================
    # Walk Forward
    # ========================================================

    def run(
        self,
        ai_data,
    ):

        data = self.prepare_data(
            ai_data
        )

        results = []

        logs = []


        # ====================================================
        # test_size と retrain_every が異なる場合でも
        # 重複予測を作らないよう
        # retrain_every単位で前進
        # ====================================================

        test_start = (
            self.initial_train_size
        )

        model_number = 0


        while test_start < len(data):

            train_end = test_start


            test_end = min(
                test_start
                + self.test_size,
                len(data),
            )


            train_data = (
                data.iloc[
                    :train_end
                ]
            )


            test_data = (
                data.iloc[
                    test_start:test_end
                ]
            )


            if test_data.empty:
                break


            X_train = train_data[
                self.feature_columns
            ]


            y_train = train_data[
                "Target"
            ]


            X_test = test_data[
                self.feature_columns
            ]


            # =================================================
            # 学習データが1クラスだけなら
            # モデルを作れないのでスキップ
            # =================================================

            if y_train.nunique() < 2:

                logs.append({

                    "Model_Number":
                        model_number + 1,

                    "Train_Start":
                        train_data.index.min(),

                    "Train_End":
                        train_data.index.max(),

                    "Test_Start":
                        test_data.index.min(),

                    "Test_End":
                        test_data.index.max(),

                    "Train_Samples":
                        len(train_data),

                    "Test_Samples":
                        len(test_data),

                    "Status":
                        "SKIPPED_ONE_CLASS",
                })

                test_start += (
                    self.retrain_every
                )

                continue


            # =================================================
            # 学習
            # =================================================

            model = (
                self.create_model()
            )


            model.fit(
                X_train,
                y_train,
            )


            model_number += 1


            # =================================================
            # 予測
            # =================================================

            probability_up = (
                self.positive_probability(
                    model,
                    X_test,
                )
            )


            prediction = (
                probability_up
                >= self.threshold
            ).astype(int)


            # =================================================
            # 結果保存
            # =================================================

            for i, date in enumerate(
                test_data.index
            ):

                row = (
                    test_data.loc[
                        date
                    ]
                )

                results.append({

                    "Date":
                        date,

                    "Probability_Up":
                        float(
                            probability_up[
                                i
                            ]
                        ),

                    "Probability_Down":
                        float(
                            1.0
                            - probability_up[
                                i
                            ]
                        ),

                    "Prediction":
                        int(
                            prediction[
                                i
                            ]
                        ),

                    "Actual":
                        int(
                            row[
                                "Target"
                            ]
                        ),

                    "Entry_Open":
                        float(
                            row[
                                "Entry_Open"
                            ]
                        ),

                    "Future_Close":
                        float(
                            row[
                                "Future_Close"
                            ]
                        ),

                    "Future_Return":
                        float(
                            row[
                                "Future_Return"
                            ]
                        ),

                    "Correct":
                        int(
                            prediction[
                                i
                            ]
                        )
                        ==
                        int(
                            row[
                                "Target"
                            ]
                        ),

                    "Target_Horizon":
                        self.target_horizon,

                    "Target_Return_Threshold":
                        self.target_return_threshold,

                    "Model_Number":
                        model_number,
                })


            logs.append({

                "Model_Number":
                    model_number,

                "Train_Start":
                    train_data.index.min(),

                "Train_End":
                    train_data.index.max(),

                "Test_Start":
                    test_data.index.min(),

                "Test_End":
                    test_data.index.max(),

                "Train_Samples":
                    len(
                        train_data
                    ),

                "Test_Samples":
                    len(
                        test_data
                    ),

                "Train_Up_Rate":
                    float(
                        y_train.mean()
                    ),

                "Status":
                    "OK",
            })


            test_start += (
                self.retrain_every
            )


        # ====================================================
        # DataFrame
        # ====================================================

        self.results = (
            pd.DataFrame(
                results
            )
        )


        self.training_log = (
            pd.DataFrame(
                logs
            )
        )


        if self.results.empty:

            raise ValueError(
                "ウォークフォワード結果が"
                "作成できませんでした。"
            )


        self.results[
            "Date"
        ] = pd.to_datetime(
            self.results[
                "Date"
            ]
        )


        # ====================================================
        # 重複除去
        # ====================================================

        self.results = (

            self.results

            .drop_duplicates(
                subset=[
                    "Date"
                ],
                keep="last",
            )

            .sort_values(
                "Date"
            )

            .set_index(
                "Date"
            )
        )


        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = (
            self.calculate_metrics()
        )


        return (
            self.results.copy(),
            dict(
                self.metrics
            ),
        )


    # ========================================================
    # Metrics
    # ========================================================

    def calculate_metrics(
        self,
    ):

        if self.results.empty:

            return {}


        actual = (
            self.results[
                "Actual"
            ].astype(int)
        )


        prediction = (
            self.results[
                "Prediction"
            ].astype(int)
        )


        probability = (
            self.results[
                "Probability_Up"
            ].astype(float)
        )


        accuracy = (
            accuracy_score(
                actual,
                prediction,
            )
        )


        precision = (
            precision_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        recall = (
            recall_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        f1 = (
            f1_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        if actual.nunique() >= 2:

            try:

                auc = (
                    roc_auc_score(
                        actual,
                        probability,
                    )
                )

            except Exception:

                auc = None

        else:

            auc = None


        matrix = confusion_matrix(
            actual,
            prediction,
            labels=[
                0,
                1,
            ],
        )


        tn = int(
            matrix[0, 0]
        )

        fp = int(
            matrix[0, 1]
        )

        fn = int(
            matrix[1, 0]
        )

        tp = int(
            matrix[1, 1]
        )


        # ====================================================
        # 高確率BUY
        # ====================================================

        high_confidence = (
            self.results[
                self.results[
                    "Probability_Up"
                ]
                >= 0.60
            ]
        )


        if high_confidence.empty:

            high_confidence_count = 0

            high_confidence_accuracy = None

            high_confidence_return = None

        else:

            high_confidence_count = int(
                len(
                    high_confidence
                )
            )

            high_confidence_accuracy = float(

                (
                    high_confidence[
                        "Actual"
                    ]
                    == 1
                ).mean()
            )

            high_confidence_return = float(

                high_confidence[
                    "Future_Return"
                ].mean()
            )


        # ====================================================
        # 予測別リターン
        # ====================================================

        predicted_up = (
            self.results[
                self.results[
                    "Prediction"
                ]
                == 1
            ]
        )


        predicted_down = (
            self.results[
                self.results[
                    "Prediction"
                ]
                == 0
            ]
        )


        avg_return_predicted_up = (

            float(
                predicted_up[
                    "Future_Return"
                ].mean()
            )

            if not predicted_up.empty

            else None
        )


        avg_return_predicted_down = (

            float(
                predicted_down[
                    "Future_Return"
                ].mean()
            )

            if not predicted_down.empty

            else None
        )


        return {

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
                    if auc is not None
                    else None
                ),

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            "actual_up_rate":
                float(
                    actual.mean()
                ),

            "predicted_up_rate":
                float(
                    prediction.mean()
                ),

            "average_future_return":
                float(
                    self.results[
                        "Future_Return"
                    ].mean()
                ),

            "average_return_predicted_up":
                avg_return_predicted_up,

            "average_return_predicted_down":
                avg_return_predicted_down,

            "high_confidence_count":
                high_confidence_count,

            "high_confidence_accuracy":
                high_confidence_accuracy,

            "high_confidence_return":
                high_confidence_return,

            "model_count":
                int(
                    self.results[
                        "Model_Number"
                    ].nunique()
                ),

            "prediction_count":
                int(
                    len(
                        self.results
                    )
                ),

            "threshold":
                self.threshold,

            # ================================================
            # Target
            # ================================================

            "target_horizon":
                self.target_horizon,

            "target_return_threshold":
                self.target_return_threshold,

            "target_definition":
                (
                    "Close(t+"
                    + str(
                        self.target_horizon
                    )
                    + ") / Open(t+1) - 1"
                ),
        }


    # ========================================================
    # Getter
    # ========================================================

    def get_results(
        self,
    ):

        return self.results.copy()


    def get_metrics(
        self,
    ):

        return dict(
            self.metrics
        )


    def get_training_log(
        self,
    ):

        return (
            self.training_log.copy()
        )


    def get_high_confidence_results(
        self,
        minimum_probability=0.60,
    ):

        if self.results.empty:

            return pd.DataFrame()

        return self.results[
            self.results[
                "Probability_Up"
            ]
            >= float(
                minimum_probability
            )
        ].copy()


# ============================================================
# Helper
# ============================================================

def run_walk_forward_backtest(
    ai_data,
    target_horizon=3,
    target_return_threshold=0.0,
    initial_train_size=500,
    test_size=20,
    retrain_every=20,
    threshold=0.50,
):

    backtest = WalkForwardBacktest(

        initial_train_size=
            initial_train_size,

        test_size=
            test_size,

        retrain_every=
            retrain_every,

        threshold=
            threshold,

        target_horizon=
            target_horizon,

        target_return_threshold=
            target_return_threshold,
    )


    return backtest.run(
        ai_data
    )
