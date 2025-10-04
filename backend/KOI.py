import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KOI:
    def __init__(self, path):
        dependent_variable = "koi_disposition"

        # Features for exoplanet classification
        independent_variables = [
            "koi_fpflag_nt",  # Not transit-like flag
            "koi_fpflag_ss",  # Stellar eclipse flag
            "koi_fpflag_co",  # Centroid offset flag
            "koi_fpflag_ec",  # Ephemeris match contamination flag
            "koi_period",  # Orbital period [days]
            "koi_duration",  # Transit duration [hrs]
            "koi_depth",  # Transit depth [ppm]!
            "koi_prad",  # Planet radius [Earth radii]!
            "koi_teq",  # Equilibrium temperature [K]!
            "koi_insol",  # Insolation flux [Earth flux]!
            "koi_model_snr",  # Transit signal-to-noise ratio!
            "koi_steff",  # Stellar effective temperature [K]!
            "koi_srad",  # Stellar radius [Solar radii]!
        ]

        df = pd.read_csv(path)

        # # Remove rows where 'koi_disposition' is "FALSE POSITIVE"
        # df = df[df['koi_disposition'] != 'FALSE POSITIVE']
        #
        # # Save the filtered CSV (optional)
        # df.to_csv('filtered_file.csv', index=False)

        # df = df.dropna(subset=independent_variables + [dependent_variable])
        x = df[independent_variables].values
        y = df[dependent_variable].values

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(x[:, 6 : len(x)])
        x[:, 6 : len(x)] = imputer.transform(x[:, 6 : len(x)])

        le = LabelEncoder()
        y = le.fit_transform(y)

        print(np.isnan(x[:, 12]).sum())

        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print(mapping)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        print("NaNs in X_train:", np.isnan(self.x_train).sum())
        print("NaNs in X_test:", np.isnan(self.x_test).sum())
        print("NaNs in y_train:", np.isnan(self.y_train).sum())

        self.sc_x = RobustScaler()
        self.x_train = self.sc_x.fit_transform(self.x_train)
        self.x_test = self.sc_x.transform(self.x_test)

        self.adaboost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,  # you can tune this
            learning_rate=0.1,
        )

        estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("ada", AdaBoostClassifier(n_estimators=50, random_state=42)),
        ]
        self.stacking = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(), cv=5
        )

        self.subspace = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=1000,
            max_features=0.5,  # subspace ratio
            max_samples=0.8,
            bootstrap=True,
            random_state=42,
        )

        self.extra_trees = ExtraTreesClassifier(
            n_estimators=200,
            max_features="sqrt",
            max_depth=None,
            criterion="entropy",
            random_state=42,
        )

        self.forest_classifier = RandomForestClassifier(
            n_estimators=50, criterion="entropy", random_state=42
        )

    def train(self):
        print("Training...")
        self.adaboost.fit(self.x_train, self.y_train)
        self.stacking.fit(self.x_train, self.y_train)
        self.forest_classifier.fit(self.x_train, self.y_train)
        self.subspace.fit(self.x_train, self.y_train)
        self.extra_trees.fit(self.x_train, self.y_train)

    def save_model(self):
        print("Saving models...")
        joblib.dump(self.adaboost, "saved_models/adaboost.joblib")
        joblib.dump(self.stacking, "saved_models/stacking.joblib")
        joblib.dump(self.forest_classifier, "saved_models/forest_classifier.joblib")
        joblib.dump(self.subspace, "saved_models/subspace.joblib")
        joblib.dump(self.extra_trees, "saved_models/extra_trees.joblib")
        joblib.dump(self.sc_x, "saved_models/scaler.joblib")

    def load_model(self):
        print("Loading models...")
        self.adaboost = joblib.load("saved_models/adaboost.joblib")
        self.stacking = joblib.load("saved_models/stacking.joblib")
        self.forest_classifier = joblib.load(
            "saved_models/forest_classifier.joblib"
        )
        self.subspace = joblib.load("saved_models/subspace.joblib")
        self.extra_trees = joblib.load("saved_models/extra_trees.joblib")
        self.sc_x = joblib.load("saved_models/scaler.joblib")

    def predict(self):
        adaboost_pred = self.adaboost.predict(self.x_test)
        stacking_pred = self.stacking.predict(self.x_test)
        forest_pred = self.forest_classifier.predict(self.x_test)
        subspace_pred = self.subspace.predict(self.x_test)
        extra_trees_pred = self.extra_trees.predict(self.x_test)

        adaboost_score = accuracy_score(self.y_test, adaboost_pred)
        stacking_score = accuracy_score(self.y_test, stacking_pred)
        forest_score = accuracy_score(self.y_test, forest_pred)
        subspace_score = accuracy_score(self.y_test, subspace_pred)
        extra_trees_score = accuracy_score(self.y_test, extra_trees_pred)
        scores = cross_val_score(
            self.forest_classifier, self.x_train, self.y_train, cv=5
        )
        print("CV mean:", scores.mean(), "CV std:", scores.std())

        print(f"(adaboost_score): {round(adaboost_score * 100, 3)}%")
        print(f"(stacking_score): {round(stacking_score * 100, 3)}%")
        print(f"(forest_score): {round(forest_score * 100, 3)}%")
        print(f"(subspace_score): {round(subspace_score * 100, 3)}%")
        print(f"(extra_trees_score): {round(extra_trees_score * 100, 3)}%")


    def get_model_statistics(self):
        """Return comprehensive model statistics for all models"""
        models = {
            "AdaBoost": self.adaboost,
            "Stacking": self.stacking,
            "Forest": self.forest_classifier,
            "Subspace": self.subspace,
            "Extra Trees": self.extra_trees,
        }

        stats = {}
        for name, model in models.items():
            pred = model.predict(self.x_test)
            stats[name] = {
                "Models": name,
                "Accuracy": round(accuracy_score(self.y_test, pred) * 100, 2),
                "Precision": round(
                    precision_score(self.y_test, pred, average="weighted") * 100, 2
                ),
                "Recall": round(
                    recall_score(self.y_test, pred, average="weighted") * 100, 2
                ),
                "F1-Score": round(
                    f1_score(self.y_test, pred, average="weighted") * 100, 2
                ),
            }

        df = pd.DataFrame(stats).T
        # Reorder columns to put Models first
        df = df[["Models", "Accuracy", "Precision", "Recall", "F1-Score"]]
        return df

    def get_feature_importance(self):
        """Get feature importance from all models that support it"""
        features = [
            "koi_fpflag_nt",
            "koi_fpflag_ss",
            "koi_fpflag_co",
            "koi_fpflag_ec",
            "koi_period",
            "koi_duration",
            "koi_depth",
            "koi_prad",
            "koi_teq",
            "koi_insol",
            "koi_model_snr",
            "koi_steff",
            "koi_srad",
        ]

        importance_data = {}

        # Forest classifier
        if hasattr(self.forest_classifier, "feature_importances_"):
            importance_data["Random Forest"] = dict(
                zip(features, self.forest_classifier.feature_importances_)
            )

        # Extra Trees
        if hasattr(self.extra_trees, "feature_importances_"):
            importance_data["Extra Trees"] = dict(
                zip(features, self.extra_trees.feature_importances_)
            )

        # AdaBoost
        if hasattr(self.adaboost, "feature_importances_"):
            importance_data["AdaBoost"] = dict(
                zip(features, self.adaboost.feature_importances_)
            )

        return pd.DataFrame(importance_data)

    def create_confusion_matrix_plot(self, model_name="Forest"):
        """Create confusion matrix plot for specified model"""
        model_map = {
            "Forest": self.forest_classifier,
            "AdaBoost": self.adaboost,
            "Stacking": self.stacking,
            "Subspace": self.subspace,
            "Extra Trees": self.extra_trees,
        }

        if model_name not in model_map:
            model_name = "Forest"

        model = model_map[model_name]
        y_pred = model.predict(self.x_test)

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        # Create plotly confusion matrix
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"{model_name} Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues",
        )

        # Add class labels
        class_labels = ["CANDIDATE", "CONFIRMED"]
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=class_labels),
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=class_labels),
        )

        return fig

    def create_model_comparison_plot(self):
        """Create comparison plot of all model performances"""
        stats_df = self.get_model_statistics()

        fig = go.Figure()

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=stats_df.index,
                    y=stats_df[metric],
                    marker_color=colors[i],
                )
            )

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score (%)",
            barmode="group",
            height=500,
        )

        return fig

    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        importance_df = self.get_feature_importance()

        if importance_df.empty:
            return None

        fig = go.Figure()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, model in enumerate(importance_df.columns):
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=importance_df.index,
                    y=importance_df[model],
                    marker_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            title="Feature Importance Across Models",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            barmode="group",
            height=500,
            xaxis_tickangle=-45,
        )

        return fig


model = KOI(path="datasets/filtered_file.csv")
# model.train()
# model.save_model()
model.load_model()
model.predict()


# Metrics
# (adaboost_score): 81.058%
# (stacking_score): 88.783%
# (forest_score): 88.571%
# (subspace_score): 88.042%
# (extra_trees_score): 87.196%
# Best parameters for forest: {'max_depth': 20, 'n_estimators': 50}
# Best cross-validated score: 86.035%

# 🎲 Running Randomized Search for adaboost...
# Fitting 5 folds for each of 30 candidates, totalling 150 fits
# Best parameters for adaboost: {'n_estimators': 1000, 'learning_rate': 0.5, 'estimator': DecisionTreeClassifier(max_depth=2)}
# Best cross-validated score for adaboost: 0.862
#
# 🎲 Running Randomized Search for forest_classifier...
# Fitting 5 folds for each of 30 candidates, totalling 150 fits
# Best parameters for forest_classifier: {'n_estimators': 300, 'max_depth': 10, 'criterion': 'entropy'}
# Best cross-validated score for forest_classifier: 0.866
#
# 🎲 Running Randomized Search for stacking...
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# /Users/eshaankhare/gitrepo/ExoVision/.venv/lib/python3.11/site-packages/sklearn/model_selection/_search.py:317: UserWarning: The total space of parameters 3 is smaller than n_iter=30. Running 3 iterations. For exhaustive searches, use GridSearchCV.
#   warnings.warn(
# Best parameters for stacking: {'final_estimator': LogisticRegression(max_iter=200), 'cv': 5}
# Best cross-validated score for stacking: 0.862
#
# 🎲 Running Randomized Search for subspace...
# Fitting 5 folds for each of 30 candidates, totalling 150 fits
# Best parameters for subspace: {'n_estimators': 1000, 'max_samples': 1.0, 'max_features': 0.7, 'estimator': DecisionTreeClassifier()}
# Best cross-validated score for subspace: 0.866
#
# 🎲 Running Randomized Search for extra_trees...
# Fitting 5 folds for each of 30 candidates, totalling 150 fits
# Best parameters for extra_trees: {'n_estimators': 200, 'max_features': None, 'max_depth': 30, 'criterion': 'gini'}
# Best cross-validated score for extra_trees: 0.867
