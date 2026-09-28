# ============================================================
# ai/model.py
#
# StockPredictionModel v2.1
#
# ・1 / 3 / 5日ターゲット統一
# ・翌営業日始値エントリー基準
# ・80 / 20 時系列分割
# ・Train/Test境界 Purge
# ・未来情報混入対策
# ・学習期間 / Test期間監査
# ・main.py v4 互換
#
# ============================================================
#
# Signal:
#   t 日終値時点
#
# Entry:
#   Open(t + 1)
#
# Evaluation:
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

MODEL_VERSION = "v2.1"


# ============================================================
# Supported target horizons
# ============================================================

SUPPORTED_HORIZONS = (
    1,
    3,
    5,
)


# ============================================================
# StockPredictionModel
# ============================================================

class StockPredictionModel:

    def __init__(
        self,

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------
        target_horizon=3,

        target_return_threshold=0.0,

        # ----------------------------------------------------
        # Train/Test
        # ----------------------------------------------------
        train_ratio=0.80,

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        prediction_threshold=0.50,

        # ----------------------------------------------------
        # RandomForest
        # ----------------------------------------------------
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
        # Target
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
                "1、3、5 のいずれかを指定してください。"
            )


        # ====================================================
        # Train/Test
        # ====================================================

        self.train_ratio = float(
            train_ratio
        )


        if not (
            0.50
            <= self.train_ratio
            < 1.0
        ):

            raise ValueError(
                "train_ratio は "
                "0.50以上1.0未満にしてください。"
            )


        # ====================================================
        # Prediction
        # ====================================================

        self.prediction_threshold = float(
            prediction_threshold
        )


        # ====================================================
        # RandomForest
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

        self.model = None

        self.feature_columns = []

        self.metrics = {}

        self.feature_importance = (
            pd.DataFrame()
        )

        self.test_results = (
            pd.DataFrame()
        )

        self.training_log = (
            pd.DataFrame()
        )

        self.prepared_data = (
            pd.DataFrame()
        )


    # ========================================================
    # Create model
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


        feature_columns = [

            column

            for column
            in requested_columns

            if column in data.columns
        ]


        # ====================================================
        # Fallback
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

                    pd.api.types.is_numeric_dtype(
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
    # Create target
    # ========================================================

    def create_target(
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
        # Entry
        # ====================================================

        data[
            "Entry_Open"
        ] = (

            data[
                "Open"
            ]
            .shift(-1)
        )


        # ====================================================
        # Evaluation price
        # ====================================================

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


        # ====================================================
        # Future return
        # ====================================================

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
        #
        # 末尾のFuture_Return=NaNを
        # 0クラスとして扱わない
        # ====================================================

        target = pd.Series(

            np.nan,

            index=data.index,

            dtype="float64",
        )


        valid_target = (

            data[
                "Future_Return"
            ]
            .notna()
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


        return data


    # ========================================================
    # Prepare training data
    # ========================================================

    def prepare_training_data(
        self,
        ai_data,
    ):

        data = (
            self.create_target(
                ai_data
            )
        )


        # ====================================================
        # Feature list
        # ====================================================

        self.feature_columns = (
            self._get_feature_columns(
                data
            )
        )


        # ====================================================
        # Numeric features
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


        required_columns = (

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
                    required_columns
            )
            .copy()
        )


        if prepared.empty:

            raise ValueError(
                "AI学習用データを作成できませんでした。"
            )


        prepared[
            "Target"
        ] = (

            prepared[
                "Target"
            ]
            .astype(int)
        )


        self.prepared_data = (
            prepared.copy()
        )


        return prepared


    # ========================================================
    # Purged chronological split
    # ========================================================

    def _purged_train_test_split(
        self,
        data,
    ):

        total_rows = len(
            data
        )


        if total_rows < 100:

            raise ValueError(
                "AI学習用データが少なすぎます。"
            )


        # ====================================================
        # Raw 80/20 boundary
        # ====================================================

        split_position = int(
            total_rows
            * self.train_ratio
        )


        if split_position <= 0:

            raise ValueError(
                "学習データを作成できません。"
            )


        if split_position >= total_rows:

            raise ValueError(
                "テストデータを作成できません。"
            )


        # ====================================================
        # PURGE
        #
        # 例:
        # target_horizon = 5
        #
        # Train末尾の5シグナルは
        # その正解ラベルの評価日が
        # Test側へ入り込む可能性があるため
        # 学習から除外する
        # ====================================================

        purged_train_end = (

            split_position

            - self.target_horizon
        )


        if purged_train_end <= 0:

            raise ValueError(
                "Purge後の学習データが不足しています。"
            )


        # ====================================================
        # Train
        # ====================================================

        train_data = (

            data.iloc[
                :purged_train_end
            ]
            .copy()
        )


        # ====================================================
        # Purged rows
        # ====================================================

        purge_data = (

            data.iloc[
                purged_train_end:
                split_position
            ]
            .copy()
        )


        # ====================================================
        # Test
        #
        # Test開始位置は元の80%境界を維持
        # ====================================================

        test_data = (

            data.iloc[
                split_position:
            ]
            .copy()
        )


        if train_data.empty:

            raise ValueError(
                "Purge後のTrainデータが空です。"
            )


        if test_data.empty:

            raise ValueError(
                "Testデータが空です。"
            )


        # ====================================================
        # Audit
        # ====================================================

        audit_record = {

            "Model_Version":
                MODEL_VERSION,

            "Target_Horizon":
                int(
                    self.target_horizon
                ),

            "Target_Return_Threshold":
                float(
                    self.target_return_threshold
                ),

            "Train_Ratio":
                float(
                    self.train_ratio
                ),

            "Total_Rows":
                int(
                    total_rows
                ),

            "Raw_Split_Position":
                int(
                    split_position
                ),

            "Purge_Rows":
                int(
                    len(
                        purge_data
                    )
                ),

            "Train_Rows":
                int(
                    len(
                        train_data
                    )
                ),

            "Test_Rows":
                int(
                    len(
                        test_data
                    )
                ),

            "Train_Start":
                train_data.index[0],

            "Train_End":
                train_data.index[-1],

            "Purge_Start":
                (
                    purge_data.index[0]

                    if not purge_data.empty

                    else pd.NaT
                ),

            "Purge_End":
                (
                    purge_data.index[-1]

                    if not purge_data.empty

                    else pd.NaT
                ),

            "Test_Start":
                test_data.index[0],

            "Test_End":
                test_data.index[-1],

            "Purged":
                True,
        }


        self.training_log = pd.DataFrame(
            [
                audit_record
            ]
        )


        return (
            train_data,
            test_data,
        )


    # ========================================================
    # Train
    # ========================================================

    def train(
        self,
        ai_data,
    ):

        data = (
            self.prepare_training_data(
                ai_data
            )
        )


        # ====================================================
        # Purged 80/20 split
        # ====================================================

        (
            train_data,
            test_data,
        ) = self._purged_train_test_split(
            data
        )


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


        X_test = (

            test_data[
                self.feature_columns
            ]
        )


        y_test = (

            test_data[
                "Target"
            ]
            .astype(int)
        )


        # ====================================================
        # Need two classes
        # ====================================================

        if y_train.nunique() < 2:

            raise ValueError(
                "学習データに0/1の両クラスがありません。"
            )


        # ====================================================
        # Fit
        # ====================================================

        self.model = (
            self._create_model()
        )


        self.model.fit(
            X_train,
            y_train,
        )


        # ====================================================
        # Probability
        # ====================================================

        probabilities = (
            self._predict_probability_from_model(
                X_test
            )
        )


        predictions = (

            probabilities

            >= self.prediction_threshold
        ).astype(int)


        # ====================================================
        # Test results
        # ====================================================

        self.test_results = pd.DataFrame(

            {

                "Actual":
                    y_test.values,

                "Prediction":
                    predictions,

                "Probability_Up":
                    probabilities,

                "Probability_Down":
                    1.0
                    - probabilities,

                "Entry_Open":
                    test_data[
                        "Entry_Open"
                    ].values,

                "Future_Close":
                    test_data[
                        "Future_Close"
                    ].values,

                "Future_Return":
                    test_data[
                        "Future_Return"
                    ].values,

            },

            index=test_data.index,
        )


        self.test_results[
            "Correct"
        ] = (

            self.test_results[
                "Actual"
            ]

            ==

            self.test_results[
                "Prediction"
            ]
        )


        self.test_results[
            "Target_Horizon"
        ] = int(
            self.target_horizon
        )


        self.test_results[
            "Target_Return_Threshold"
        ] = float(
            self.target_return_threshold
        )


        self.test_results[
            "Model_Version"
        ] = MODEL_VERSION


        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = (
            self._calculate_metrics(
                y_test=
                    y_test,

                predictions=
                    predictions,

                probabilities=
                    probabilities,

                test_data=
                    test_data,

                train_data=
                    train_data,
            )
        )


        # ====================================================
        # Feature importance
        # ====================================================

        self.feature_importance = pd.DataFrame(

            {

                "Feature":
                    self.feature_columns,

                "Importance":
                    self.model
                    .feature_importances_,
            }
        )


        self.feature_importance = (

            self.feature_importance
            .sort_values(
                "Importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        return dict(
            self.metrics
        )


    # ========================================================
    # Internal probability
    # ========================================================

    def _predict_probability_from_model(
        self,
        features,
    ):

        if self.model is None:

            raise ValueError(
                "AIモデルがまだ学習されていません。"
            )


        probabilities = (
            self.model.predict_proba(
                features
            )
        )


        classes = list(
            self.model.classes_
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


        return np.zeros(
            len(
                features
            ),
            dtype=float,
        )


    # ========================================================
    # Predict probability
    #
    # main.py v4 compatibility
    # ========================================================

    def predict_probability(
        self,
        ai_data,
    ):

        if self.model is None:

            raise ValueError(
                "AIモデルがまだ学習されていません。"
            )


        data = (
            self._normalize_dataframe(
                ai_data
            )
        )


        if data.empty:

            raise ValueError(
                "予測対象データが空です。"
            )


        missing_features = [

            column

            for column
            in self.feature_columns

            if column not in data.columns
        ]


        if missing_features:

            raise ValueError(

                "予測に必要な特徴量がありません: "

                + ", ".join(
                    missing_features
                )
            )


        # ====================================================
        # Numeric
        # ====================================================

        feature_data = (

            data[
                self.feature_columns
            ]
            .copy()
        )


        for column in self.feature_columns:

            feature_data[
                column
            ] = pd.to_numeric(
                feature_data[
                    column
                ],
                errors="coerce",
            )


        feature_data.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
            inplace=True,
        )


        # ====================================================
        # 最新の完全な特徴量行を使用
        # ====================================================

        valid_features = (
            feature_data
            .dropna()
        )


        if valid_features.empty:

            raise ValueError(
                "最新予測に使用できる特徴量がありません。"
            )


        latest_features = (

            valid_features
            .iloc[
                [
                    -1
                ]
            ]
        )


        probability = (
            self._predict_probability_from_model(
                latest_features
            )
        )


        return float(
            probability[
                0
            ]
        )


    # ========================================================
    # Predict
    # ========================================================

    def predict(
        self,
        ai_data,
    ):

        probability = (
            self.predict_probability(
                ai_data
            )
        )


        return int(
            probability
            >= self.prediction_threshold
        )


    # ========================================================
    # Metrics
    # ========================================================

    def _calculate_metrics(
        self,
        y_test,
        predictions,
        probabilities,
        test_data,
        train_data,
    ):

        actual = (
            pd.Series(
                y_test
            )
            .astype(int)
        )


        predicted = pd.Series(

            predictions,

            index=actual.index,

            dtype=int,
        )


        probability_series = pd.Series(

            probabilities,

            index=actual.index,

            dtype=float,
        )


        # ====================================================
        # Classification
        # ====================================================

        accuracy = float(
            accuracy_score(
                actual,
                predicted,
            )
        )


        precision = float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        )


        recall = float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        )


        f1 = float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        )


        # ====================================================
        # AUC
        # ====================================================

        if actual.nunique() >= 2:

            try:

                auc = float(
                    roc_auc_score(
                        actual,
                        probability_series,
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

            predicted,

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
        # Future return
        # ====================================================

        future_return = pd.to_numeric(

            test_data[
                "Future_Return"
            ],

            errors="coerce",
        )


        predicted_up_mask = (

            predicted.values == 1
        )


        predicted_down_mask = (

            predicted.values == 0
        )


        predicted_up_returns = (

            future_return.iloc[
                np.where(
                    predicted_up_mask
                )[0]
            ]
        )


        predicted_down_returns = (

            future_return.iloc[
                np.where(
                    predicted_down_mask
                )[0]
            ]
        )


        # ====================================================
        # Audit information
        # ====================================================

        audit = {}


        if not self.training_log.empty:

            audit = (
                self.training_log
                .iloc[
                    0
                ]
                .to_dict()
            )


        # ====================================================
        # Metrics
        # ====================================================

        return {

            # ------------------------------------------------
            # Version
            # ------------------------------------------------

            "model_version":
                MODEL_VERSION,

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            "train_ratio":
                float(
                    self.train_ratio
                ),

            "purged":
                True,

            "purge_days":
                int(
                    self.target_horizon
                ),

            "train_rows":
                int(
                    len(
                        train_data
                    )
                ),

            "test_rows":
                int(
                    len(
                        test_data
                    )
                ),

            "purge_rows":
                int(
                    audit.get(
                        "Purge_Rows",
                        self.target_horizon,
                    )
                ),

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Confusion
            # ------------------------------------------------

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            "actual_up_count":
                int(
                    (
                        actual == 1
                    ).sum()
                ),

            "actual_down_count":
                int(
                    (
                        actual == 0
                    ).sum()
                ),

            "predicted_up_count":
                int(
                    (
                        predicted == 1
                    ).sum()
                ),

            "predicted_down_count":
                int(
                    (
                        predicted == 0
                    ).sum()
                ),

            # ------------------------------------------------
            # Return
            # ------------------------------------------------

            "average_future_return":
                (
                    float(
                        future_return.mean()
                    )

                    if future_return.notna().any()

                    else None
                ),

            "average_return_predicted_up":
                (
                    float(
                        predicted_up_returns.mean()
                    )

                    if (
                        predicted_up_returns
                        .notna()
                        .any()
                    )

                    else None
                ),

            "average_return_predicted_down":
                (
                    float(
                        predicted_down_returns.mean()
                    )

                    if (
                        predicted_down_returns
                        .notna()
                        .any()
                    )

                    else None
                ),
        }


    # ========================================================
    # Feature importance
    # ========================================================

    def get_feature_importance(
        self,
    ):

        return self.feature_importance.copy()


    # ========================================================
    # Test results
    # ========================================================

    def get_test_results(
        self,
    ):

        return self.test_results.copy()


    # ========================================================
    # Training audit
    # ========================================================

    def get_training_log(
        self,
    ):

        return self.training_log.copy()


    # ========================================================
    # Metrics
    # ========================================================

    def get_metrics(
        self,
    ):

        return dict(
            self.metrics
        )


    # ========================================================
    # Target info
    # ========================================================

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

            "future_return_definition":
                (
                    "Close(t+horizon) / "
                    "Open(t+1) - 1"
                ),

            "supported_horizons":
                list(
                    SUPPORTED_HORIZONS
                ),

            "model_version":
                MODEL_VERSION,

            "purged":
                True,

            "purge_days":
                int(
                    self.target_horizon
                ),
        }


    # ========================================================
    # Model info
    # ========================================================

    def get_model_info(
        self,
    ):

        return {

            "model":
                "RandomForestClassifier",

            "version":
                MODEL_VERSION,

            "target_horizon":
                int(
                    self.target_horizon
                ),

            "target_return_threshold":
                float(
                    self.target_return_threshold
                ),

            "train_ratio":
                float(
                    self.train_ratio
                ),

            "prediction_threshold":
                float(
                    self.prediction_threshold
                ),

            "purged":
                True,

            "purge_days":
                int(
                    self.target_horizon
                ),

            "n_estimators":
                int(
                    self.n_estimators
                ),

            "max_depth":
                self.max_depth,

            "min_samples_split":
                int(
                    self.min_samples_split
                ),

            "min_samples_leaf":
                int(
                    self.min_samples_leaf
                ),

            "max_features":
                self.max_features,

            "random_state":
                int(
                    self.random_state
                ),

            "class_weight":
                self.class_weight,

            "feature_count":
                int(
                    len(
                        self.feature_columns
                    )
                ),
        }
