# ============================================================
# backtest/engine.py
#
# WalkForwardBacktest v2.1
#
# main.py v4 対応
# ai/model.py v2 対応
#
# ------------------------------------------------------------
# AIターゲット定義
#
# シグナル日:
#   t
#
# エントリー想定:
#   Open(t + 1)
#
# 評価価格:
#   Close(t + target_horizon)
#
# Future_Return:
#   Close(t + target_horizon)
#   / Open(t + 1)
#   - 1
#
# Target:
#   Future_Return > target_return_threshold
#
# target_horizon:
#   1 / 3 / 5
#
# ------------------------------------------------------------
# v2.1
#
# ・1日 / 3日 / 5日ターゲット対応
# ・target_return_threshold対応
# ・Walk-Forward
# ・Expanding Window
# ・Purge処理
# ・未来情報混入対策
# ・Probability_Up出力
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


# ============================================================
# Version
# ============================================================

BACKTEST_ENGINE_VERSION = "v2.1"


# ============================================================
# Supported target horizons
# ============================================================

SUPPORTED_HORIZONS = (
    1,
    3,
    5,
)


# ============================================================
# WalkForwardBacktest
# ============================================================

class WalkForwardBacktest:

    def __init__(
        self,

        initial_train_size=500,

        test_size=20,

        retrain_every=20,

        prediction_threshold=0.50,

        target_horizon=3,

        target_return_threshold=0.0,

        n_estimators=500,

        max_depth=8,

        min_samples_split=10,

        min_samples_leaf=5,

        max_features="sqrt",

        random_state=42,

        class_weight="balanced",

        n_jobs=-1,
    ):

        # ====================================================
        # Walk-forward settings
        # ====================================================

        self.initial_train_size = int(
            initial_train_size
        )

        self.test_size = int(
            test_size
        )

        self.retrain_every = int(
            retrain_every
        )

        self.prediction_threshold = float(
            prediction_threshold
        )


        # ====================================================
        # Target settings
        # ====================================================

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
                "1, 3, 5 のいずれかを指定してください。"
            )


        # ====================================================
        # Model settings
        # ====================================================

        self.n_estimators = int(
            n_estimators
        )

        self.max_depth = max_depth

        self.min_samples_split = int(
            min_samples_split
        )

        self.min_samples_leaf = int(
            min_samples_leaf
        )

        self.max_features = max_features

        self.random_state = int(
            random_state
        )

        self.class_weight = class_weight

        self.n_jobs = int(
            n_jobs
        )


        # ====================================================
        # Runtime
        # ====================================================

        self.feature_columns = []

        self.results = pd.DataFrame()

        self.metrics = {}

        self.model_count = 0


    # ========================================================
    # Model factory
    # ========================================================

    def _create_model(
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
                self.class_weight,

            n_jobs=
                self.n_jobs,
        )


    # ========================================================
    # Normalize dataframe
    # ========================================================

    @staticmethod
    def _normalize_dataframe(
        data,
    ):

        if data is None:

            return pd.DataFrame()


        result = data.copy()


        if result.empty:

            return result


        result.index = pd.to_datetime(
            result.index
        )


        if getattr(
            result.index,
            "tz",
            None,
        ) is not None:

            result.index = (
                result.index
                .tz_localize(None)
            )


        result = result[
            ~result.index.duplicated(
                keep="last"
            )
        ]


        result = result.sort_index()


        return result


    # ========================================================
    # Feature columns
    # ========================================================

    def _get_feature_columns(
        self,
        data,
    ):

        try:

            requested_columns = (
                get_ai_feature_columns()
            )

        except Exception:

            requested_columns = []


        # ====================================================
        # features.py のリストを優先
        # ====================================================

        feature_columns = [

            column

            for column
            in requested_columns

            if column in data.columns
        ]


        # ====================================================
        # 万一取得できなかった場合
        # 数値列からTarget関連を除外
        # ====================================================

        if not feature_columns:

            excluded_columns = {

                "Target",

                "Future_Return",

                "Entry_Open",

                "Future_Close",
            }


            feature_columns = [

                column

                for column
                in data.columns

                if (
                    column
                    not in excluded_columns

                    and

                    pd.api.types
                    .is_numeric_dtype(
                        data[column]
                    )
                )
            ]


        if not feature_columns:

            raise ValueError(
                "AI学習に使用できる特徴量がありません。"
            )


        return feature_columns


    # ========================================================
    # Prepare target
    # ========================================================

    def prepare_data(
        self,
        ai_data,
    ):

        data = (
            self._normalize_dataframe(
                ai_data
            )
        )


        if data.empty:

            raise ValueError(
                "ai_data が空です。"
            )


        # ====================================================
        # 必須列
        # ====================================================

        required_columns = [
            "Open",
            "Close",
        ]


        missing_columns = [

            column

            for column
            in required_columns

            if column not in data.columns
        ]


        if missing_columns:

            raise ValueError(
                "ai_data に必要な列がありません: "
                + ", ".join(
                    missing_columns
                )
            )


        # ====================================================
        # Numeric
        # ====================================================

        data[
            "Open"
        ] = pd.to_numeric(
            data[
                "Open"
            ],
            errors="coerce",
        )


        data[
            "Close"
        ] = pd.to_numeric(
            data[
                "Close"
            ],
            errors="coerce",
        )


        # ====================================================
        # Target
        #
        # Signal = t close
        # Entry  = Open(t+1)
        # Exit評価 = Close(t+horizon)
        # ====================================================

        data[
            "Entry_Open"
        ] = (

            data[
                "Open"
            ]
            .shift(-1)
        )


        data[
            "Future_Close"
        ] = (

            data[
                "Close"
            ]
            .shift(
                -self.target_horizon
            )
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
        # TargetはNaNを先に維持
        #
        # Future_Returnがない末尾を
        # 誤って0クラスにしない
        # ====================================================

        target = pd.Series(
            np.nan,
            index=data.index,
            dtype="float64",
        )


        valid_target = (
            data[
                "Future_Return"
            ].notna()
        )


        target.loc[
            valid_target
        ] = (

            data.loc[
                valid_target,
                "Future_Return",
            ]

            >

            self.target_return_threshold
        ).astype(int)


        data[
            "Target"
        ] = target


        # ====================================================
        # Feature columns
        # ====================================================

        self.feature_columns = (
            self._get_feature_columns(
                data
            )
        )


        # ====================================================
        # Numeric conversion
        # ====================================================

        for column in self.feature_columns:

            data[
                column
            ] = pd.to_numeric(
                data[
                    column
                ],
                errors="coerce",
            )


        # ====================================================
        # inf -> NaN
        # ====================================================

        data.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
            inplace=True,
        )


        # ====================================================
        # 特徴量が揃っている行だけ残す
        #
        # Target NaNの最新行は
        # Walk-forward評価には使わないため
        # ここではTargetも必須
        # ====================================================

        required_for_backtest = (

            self.feature_columns

            + [
                "Entry_Open",
                "Future_Close",
                "Future_Return",
                "Target",
            ]
        )


        prepared = (
            data
            .dropna(
                subset=
                    required_for_backtest
            )
            .copy()
        )


        prepared[
            "Target"
        ] = (

            prepared[
                "Target"
            ]
            .astype(int)
        )


        if prepared.empty:

            raise ValueError(
                "ターゲット作成後のデータが空です。"
            )


        return prepared


    # ========================================================
    # Purged training data
    # ========================================================

    def _get_purged_training_data(
        self,
        data,
        test_start_position,
    ):

        # ====================================================
        # 重要:
        #
        # test_start_position の日の予測を行う時点では、
        # 直近 target_horizon 日分のTarget結果は
        # まだ確定していない可能性があります。
        #
        # 例:
        # target_horizon = 3
        #
        # test開始 = t
        #
        # t-1 のTargetは
        # Close(t+2)を必要とするため
        # t時点では未知。
        #
        # したがって、
        #
        # train_end =
        # test_start_position - target_horizon
        #
        # とします。
        # ====================================================

        train_end_position = (

            test_start_position

            - self.target_horizon
        )


        if train_end_position <= 0:

            return pd.DataFrame()


        train_data = (
            data.iloc[
                :train_end_position
            ]
            .copy()
        )


        return train_data


    # ========================================================
    # Probability helper
    # ========================================================

    @staticmethod
    def _probability_up(
        model,
        features,
    ):

        probabilities = (
            model.predict_proba(
                features
            )
        )


        classes = list(
            model.classes_
        )


        if 1 in classes:

            class_index = (
                classes.index(
                    1
                )
            )


            return probabilities[
                :,
                class_index
            ]


        # ====================================================
        # 学習データが全て0の場合
        # ====================================================

        return np.zeros(
            len(
                features
            ),
            dtype=float,
        )


    # ========================================================
    # Walk-forward
    # ========================================================

    def run(
        self,
        ai_data,
    ):

        data = (
            self.prepare_data(
                ai_data
            )
        )


        total_rows = len(
            data
        )


        # ====================================================
        # 必要データ量チェック
        # ====================================================

        minimum_required = (

            self.initial_train_size

            + self.target_horizon

            + 1
        )


        if (
            total_rows
            < minimum_required
        ):

            raise ValueError(

                "ウォークフォワード検証に必要な"
                "データ数が不足しています。"

                f" 現在: {total_rows} 行 /"

                f" 必要: {minimum_required} 行以上"
            )


        # ====================================================
        # Walk-forward
        # ====================================================

        result_records = []

        model = None

        last_train_position = None

        model_number = 0


        # ====================================================
        # test開始位置
        #
        # purge後でも initial_train_size を
        # 確保するため horizon を追加
        # ====================================================

        test_start = (

            self.initial_train_size

            + self.target_horizon
        )


        while test_start < total_rows:

            test_end = min(

                test_start
                + self.test_size,

                total_rows,
            )


            # =================================================
            # Purged training set
            # =================================================

            train_data = (
                self._get_purged_training_data(

                    data=data,

                    test_start_position=
                        test_start,
                )
            )


            if (
                len(train_data)
                < self.initial_train_size
            ):

                test_start = test_end

                continue


            # =================================================
            # Retrain
            # =================================================

            should_retrain = (

                model is None

                or

                last_train_position
                is None

                or

                (
                    test_start
                    - last_train_position
                )
                >= self.retrain_every
            )


            if should_retrain:

                X_train = (
                    train_data[
                        self.feature_columns
                    ]
                )


                y_train = (
                    train_data[
                        "Target"
                    ]
                    .astype(int)
                )


                # =============================================
                # RandomForestは2クラスが理想
                # =============================================

                if (
                    y_train.nunique()
                    < 2
                ):

                    # -----------------------------------------
                    # クラスが1種類しかない場合は
                    # この期間をスキップ
                    # -----------------------------------------

                    test_start = test_end

                    continue


                model = (
                    self._create_model()
                )


                model.fit(
                    X_train,
                    y_train,
                )


                model_number += 1

                last_train_position = (
                    test_start
                )


            # =================================================
            # Test block
            # =================================================

            test_data = (
                data.iloc[
                    test_start:test_end
                ]
                .copy()
            )


            if test_data.empty:

                break


            X_test = (
                test_data[
                    self.feature_columns
                ]
            )


            probabilities_up = (
                self._probability_up(
                    model,
                    X_test,
                )
            )


            predictions = (

                probabilities_up

                >= self.prediction_threshold
            ).astype(int)


            # =================================================
            # Save results
            # =================================================

            for row_number, (
                date,
                row,
            ) in enumerate(
                test_data.iterrows()
            ):

                probability_up = float(
                    probabilities_up[
                        row_number
                    ]
                )


                prediction = int(
                    predictions[
                        row_number
                    ]
                )


                actual = int(
                    row[
                        "Target"
                    ]
                )


                result_records.append({

                    "Date":
                        date,

                    "Probability_Up":
                        probability_up,

                    "Probability_Down":
                        float(
                            1.0
                            - probability_up
                        ),

                    "Prediction":
                        prediction,

                    "Actual":
                        actual,

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
                        bool(
                            prediction
                            == actual
                        ),

                    "Target_Horizon":
                        int(
                            self.target_horizon
                        ),

                    "Target_Return_Threshold":
                        float(
                            self.target_return_threshold
                        ),

                    "Model_Number":
                        int(
                            model_number
                        ),

                    "Engine_Version":
                        BACKTEST_ENGINE_VERSION,
                })


            # =================================================
            # Next block
            # =================================================

            test_start = test_end


        # ====================================================
        # Results
        # ====================================================

        self.results = pd.DataFrame(
            result_records
        )


        if self.results.empty:

            raise ValueError(
                "ウォークフォワード検証結果が"
                "作成できませんでした。"
            )


        self.results[
            "Date"
        ] = pd.to_datetime(
            self.results[
                "Date"
            ]
        )


        self.results = (

            self.results
            .drop_duplicates(
                subset=[
                    "Date"
                ],
                keep="last",
            )
            .set_index(
                "Date"
            )
            .sort_index()
        )


        self.model_count = int(
            model_number
        )


        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = (
            self._calculate_metrics(
                self.results
            )
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

    def _calculate_metrics(
        self,
        results,
    ):

        if results.empty:

            return {}


        actual = (
            pd.to_numeric(
                results[
                    "Actual"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )


        prediction = (
            pd.to_numeric(
                results[
                    "Prediction"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )


        probability = (
            pd.to_numeric(
                results[
                    "Probability_Up"
                ],
                errors="coerce",
            )
        )


        future_return = (
            pd.to_numeric(
                results[
                    "Future_Return"
                ],
                errors="coerce",
            )
        )


        # ====================================================
        # Standard metrics
        # ====================================================

        accuracy = float(
            accuracy_score(
                actual,
                prediction,
            )
        )


        precision = float(
            precision_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        recall = float(
            recall_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        f1 = float(
            f1_score(
                actual,
                prediction,
                zero_division=0,
            )
        )


        # ====================================================
        # AUC
        # ====================================================

        if (
            actual.nunique()
            >= 2

            and

            probability.notna().any()
        ):

            try:

                auc = float(
                    roc_auc_score(
                        actual,
                        probability,
                    )
                )

            except Exception:

                auc = None

        else:

            auc = None


        # ====================================================
        # Confusion matrix
        # ====================================================

        matrix = confusion_matrix(
            actual,
            prediction,
            labels=[
                0,
                1,
            ],
        )


        tn = int(
            matrix[
                0,
                0
            ]
        )

        fp = int(
            matrix[
                0,
                1
            ]
        )

        fn = int(
            matrix[
                1,
                0
            ]
        )

        tp = int(
            matrix[
                1,
                1
            ]
        )


        # ====================================================
        # Counts
        # ====================================================

        prediction_count = int(
            len(
                results
            )
        )


        actual_up_count = int(
            (
                actual == 1
            ).sum()
        )


        actual_down_count = int(
            (
                actual == 0
            ).sum()
        )


        predicted_up_count = int(
            (
                prediction == 1
            ).sum()
        )


        predicted_down_count = int(
            (
                prediction == 0
            ).sum()
        )


        # ====================================================
        # Return statistics
        # ====================================================

        average_future_return = (

            float(
                future_return.mean()
            )

            if future_return.notna().any()

            else None
        )


        predicted_up_returns = (
            future_return[
                prediction == 1
            ]
        )


        predicted_down_returns = (
            future_return[
                prediction == 0
            ]
        )


        average_return_predicted_up = (

            float(
                predicted_up_returns.mean()
            )

            if predicted_up_returns
            .notna()
            .any()

            else None
        )


        average_return_predicted_down = (

            float(
                predicted_down_returns.mean()
            )

            if predicted_down_returns
            .notna()
            .any()

            else None
        )


        # ====================================================
        # High confidence
        #
        # 0.60以上を上昇高信頼として参考表示
        # ====================================================

        high_confidence_mask = (

            probability
            >= 0.60
        )


        high_confidence_count = int(
            high_confidence_mask.sum()
        )


        if high_confidence_count > 0:

            high_confidence_accuracy = float(

                (
                    actual[
                        high_confidence_mask
                    ]

                    ==

                    prediction[
                        high_confidence_mask
                    ]
                ).mean()
            )


            high_confidence_avg_return = float(

                future_return[
                    high_confidence_mask
                ].mean()
            )

        else:

            high_confidence_accuracy = None

            high_confidence_avg_return = None


        # ====================================================
        # Metrics dictionary
        # ====================================================

        return {

            # ================================================
            # Version
            # ================================================

            "engine_version":
                BACKTEST_ENGINE_VERSION,

            # ================================================
            # Target
            # ================================================

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),

            "target_definition":
                (
                    "Close(t+horizon) / "
                    "Open(t+1) - 1"
                ),

            "purge_days":
                int(
                    self.target_horizon
                ),

            # ================================================
            # Model / predictions
            # ================================================

            "model_count":
                int(
                    self.model_count
                ),

            "prediction_count":
                prediction_count,

            # ================================================
            # Classification
            # ================================================

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "auc":
                auc,

            # ================================================
            # Confusion matrix
            # ================================================

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            # ================================================
            # Counts
            # ================================================

            "actual_up_count":
                actual_up_count,

            "actual_down_count":
                actual_down_count,

            "predicted_up_count":
                predicted_up_count,

            "predicted_down_count":
                predicted_down_count,

            # ================================================
            # Return
            # ================================================

            "average_future_return":
                average_future_return,

            "average_return_predicted_up":
                average_return_predicted_up,

            "average_return_predicted_down":
                average_return_predicted_down,

            # ================================================
            # High confidence
            # ================================================

            "high_confidence_count":
                high_confidence_count,

            "high_confidence_accuracy":
                high_confidence_accuracy,

            "high_confidence_avg_return":
                high_confidence_avg_return,
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


    def get_target_info(
        self,
    ):

        return {

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),

            "entry_price":
                "Open(t+1)",

            "evaluation_price":
                (
                    f"Close(t+"
                    f"{self.target_horizon})"
                ),

            "target_definition":
                (
                    "Future_Return > "
                    "target_return_threshold"
                ),

            "purge_days":
                int(
                    self.target_horizon
                ),

            "engine_version":
                BACKTEST_ENGINE_VERSION,
        }


    def get_engine_info(
        self,
    ):

        return {

            "engine":
                "WalkForwardBacktest",

            "version":
                BACKTEST_ENGINE_VERSION,

            "method":
                "Expanding Walk-Forward",

            "purged":
                True,

            "purge_days":
                int(
                    self.target_horizon
                ),

            "initial_train_size":
                int(
                    self.initial_train_size
                ),

            "test_size":
                int(
                    self.test_size
                ),

            "retrain_every":
                int(
                    self.retrain_every
                ),

            "prediction_threshold":
                float(
                    self.prediction_threshold
                ),

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),
        }
