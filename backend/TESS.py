import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Dependent variable (target)
dependent_variable = "disposition"

# Independent variables (features)
independent_variables = [
    "pl_orbper",  # Orbital Period [days]
    "pl_orbsmax",  # Semi-Major Axis [au]
    "pl_rade",  # Planet Radius [Earth radii]
    "pl_bmasse",  # Planet Mass [Earth masses]
    "pl_insol",  # Insolation Flux [Earth flux]
    "pl_eqt",  # Equilibrium Temperature [K]
    "st_teff",  # Stellar Effective Temperature [K]
    "st_rad",  # Stellar Radius [Solar radii]
    "st_mass",  # Stellar Mass [Solar masses]
    "st_met",  # Stellar Metallicity [dex]
    "st_logg"  # Stellar Surface Gravity [log10(cm/s²)]
]


# df = pd.read_csv('datasets/k2pandc_2025.09.19_14.29.24.csv')
#
# # Remove rows where 'disposition' is "FALSE POSITIVE"
# df = df[df['disposition'] != 'FALSE POSITIVE']
#
#
# # Keep only the selected columns
# df_filtered = df[[dependent_variable] + independent_variables]
#
# # Save to CSV
# df_filtered.to_csv("datasets/filtered_features_k2pandc.csv", index=False)


class TESS:
    def __init__(self, path):
        df = pd.read_csv(path)

        x = df[independent_variables].values

        y = df[dependent_variable].values

        le = LabelEncoder()
        y = le.fit_transform(y)

        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print(mapping)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(x)

        x = imputer.transform(x)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        sc = RobustScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        self.adaboost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,  # you can tune this
            learning_rate=0.1
        )

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        self.stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )

        self.subspace = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=1000,
            max_features=0.5,  # subspace ratio
            max_samples=0.8,
            bootstrap=True,
            random_state=42
        )

        self.extra_trees = ExtraTreesClassifier(
            n_estimators=200,
            max_features='sqrt',
            max_depth=None,
            criterion='entropy',
            random_state=42
        )

        self.forest_classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)

    def train(self):
        print('Training...')
        self.adaboost.fit(self.x_train, self.y_train)
        self.stacking.fit(self.x_train, self.y_train)
        self.forest_classifier.fit(self.x_train, self.y_train)
        self.subspace.fit(self.x_train, self.y_train)
        self.extra_trees.fit(self.x_train, self.y_train)

    def save_model(self):
        print('Saving models...')
        joblib.dump(self.adaboost, 'saved_models/TESS/adaboost.joblib')
        joblib.dump(self.stacking, 'saved_models/TESS/stacking.joblib')
        joblib.dump(self.forest_classifier, 'saved_models/TESS/forest_classifier.joblib')
        joblib.dump(self.subspace, 'saved_models/TESS/subspace.joblib')
        joblib.dump(self.extra_trees, 'saved_models/TESS/extra_trees.joblib')

    def load_model(self):
        print('Loading models...')
        import os
        model_dir = os.path.join(os.path.dirname(__file__), 'saved_models', 'TESS')
        self.adaboost = joblib.load(os.path.join(model_dir, 'adaboost.joblib'))
        self.stacking = joblib.load(os.path.join(model_dir, 'stacking.joblib'))
        self.forest_classifier = joblib.load(os.path.join(model_dir, 'forest_classifier.joblib'))
        self.subspace = joblib.load(os.path.join(model_dir, 'subspace.joblib'))
        self.extra_trees = joblib.load(os.path.join(model_dir, 'extra_trees.joblib'))

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
        scores = cross_val_score(self.forest_classifier, self.x_train, self.y_train, cv=5)
        print("CV mean:", scores.mean(), "CV std:", scores.std())

        print(f'(adaboost_score): {round(adaboost_score * 100, 3)}%')
        print(f'(stacking_score): {round(stacking_score * 100, 3)}%')
        print(f'(forest_score): {round(forest_score * 100, 3)}%')
        print(f'(subspace_score): {round(subspace_score * 100, 3)}%')
        print(f'(extra_trees_score): {round(extra_trees_score * 100, 3)}%')


# Uncomment the lines below to train and save models
# model = TESS('datasets/filtered_features_k2pandc.csv')
# model.train()
# model.load_model()
# model.predict()
