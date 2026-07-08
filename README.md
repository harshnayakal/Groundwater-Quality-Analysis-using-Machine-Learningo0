# Groundwater Quality Analysis using Machine Learning

## Overview

This project performs groundwater quality analysis using Machine Learning techniques. The system preprocesses the dataset, performs exploratory data analysis, applies feature scaling, dimensionality reduction, clustering, and predictive modeling to evaluate groundwater quality.

---

## Features

- Data Cleaning
- Missing Value Detection
- Missing Value Imputation
- Duplicate Removal
- Data Normalization
- Statistical Summary
- Correlation Analysis
- Visualization Dashboard
- PCA Analysis
- K-Means Clustering
- Machine Learning Prediction
- Feature Importance Analysis
- Automatic Report Generation

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Machine Learning Algorithms

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression
- Principal Component Analysis (PCA)
- K-Means Clustering

---

## Data Preprocessing

- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Standardization
- Feature Scaling

---

## Code

# ==========================================================
# DATA OPTIMIZATION
# ==========================================================
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import numpy as np

class DataOptimization:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # ======================================================
    # Dataset Summary
    # ======================================================
    def dataset_summary(self):

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)

        print(self.df.describe())

    # ======================================================
    # Missing Values
    # ======================================================
    def missing_values(self):

        print("\n" + "="*60)
        print("MISSING VALUES")
        print("="*60)

        print(self.df.isnull().sum())

    # ======================================================
    # Missing Value Heatmap
    # ======================================================
    def missing_value_heatmap(self):

        plt.figure(figsize=(14,6))

        sns.heatmap(
            self.df.isnull(),
            cbar=False,
            cmap="viridis"
        )

        plt.title("Missing Value Heatmap")

        plt.tight_layout()

        plt.savefig("Missing_Value_Heatmap.png", dpi=300)

        plt.show()

    # ======================================================
    # Duplicate Records
    # ======================================================
    def duplicate_records(self):

        duplicates = self.df.duplicated().sum()

        print("\nDuplicate Records :", duplicates)

    # ======================================================
    # Remove Duplicates
    # ======================================================
    def remove_duplicates(self):

        before = len(self.df)

        self.df = self.df.drop_duplicates()

        after = len(self.df)

        print("\nRows Before :", before)

        print("Rows After  :", after)

    # ======================================================
    # Fill Missing Values
    # ======================================================
    def fill_missing(self):

        numeric = self.df.select_dtypes(include=np.number).columns

        imputer = SimpleImputer(strategy="mean")

        self.df[numeric] = imputer.fit_transform(
            self.df[numeric]
        )

        print("\nMissing Values Filled Successfully.")

    # ======================================================
    # Correlation Heatmap
    # ======================================================
    def correlation(self):

        plt.figure(figsize=(18,12))

        corr = self.df.select_dtypes(include=np.number).corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title(
            "Groundwater Parameter Correlation Matrix",
            fontsize=18,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig("Correlation_Heatmap.png", dpi=300)

        plt.show()

    # ======================================================
    # Histogram Dashboard
    # ======================================================
    def histogram_dashboard(self):

        numeric = self.df.select_dtypes(include=np.number)

        numeric.hist(
            figsize=(20,16),
            bins=15
        )

        plt.suptitle(
            "Distribution of Groundwater Parameters",
            fontsize=18,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig("Histogram_Dashboard.png", dpi=300)

        plt.show()

    # ======================================================
    # Boxplot Dashboard
    # ======================================================
    def boxplot_dashboard(self):

        numeric = self.df.select_dtypes(include=np.number)

        plt.figure(figsize=(20,8))

        numeric.boxplot(rot=90)

        plt.title(
            "Outlier Detection using Boxplots",
            fontsize=18,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig("Boxplot_Dashboard.png", dpi=300)

        plt.show()

    # ======================================================
    # Normalize Dataset
    # ======================================================
    def normalize(self):

        scaler = StandardScaler()

        numeric = self.df.select_dtypes(include=np.number)

        scaled = scaler.fit_transform(numeric)

        scaled_df = pd.DataFrame(
            scaled,
            columns=numeric.columns
        )

        print("\nNormalized Dataset")

        print(scaled_df.head())

        scaled_df.to_csv(
            "Optimized_Dataset.csv",
            index=False
        )

        return scaled_df

# ==========================================================
# PCA AND CLUSTERING
# ==========================================================

class MachineLearningAnalysis:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    # ======================================================
    # PCA Analysis
    # ======================================================

    def perform_pca(self):

        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)

        pca = PCA(n_components=2)

        pca_result = pca.fit_transform(self.df)

        print("\nExplained Variance Ratio")

        print(pca.explained_variance_ratio_)

        plt.figure(figsize=(10,7))

        plt.scatter(
            pca_result[:,0],
            pca_result[:,1]
        )

        plt.title("PCA Projection")

        plt.xlabel("Principal Component 1")

        plt.ylabel("Principal Component 2")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig("PCA_Projection.png", dpi=300)

        plt.show()

        return pca_result

    # ======================================================
    # Explained Variance Graph
    # ======================================================

    def explained_variance(self):

        pca = PCA()

        pca.fit(self.df)

        variance = pca.explained_variance_ratio_

        plt.figure(figsize=(10,6))

        plt.plot(
            range(1, len(variance)+1),
            variance,
            marker='o'
        )

        plt.title("Explained Variance Ratio")

        plt.xlabel("Principal Components")

        plt.ylabel("Variance")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig("Explained_Variance.png", dpi=300)

        plt.show()

    # ======================================================
    # Elbow Method
    # ======================================================

    def elbow_method(self):

        inertia = []

        K = range(1,11)

        for k in K:

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            model.fit(self.df)

            inertia.append(model.inertia_)

        plt.figure(figsize=(8,6))

        plt.plot(
            K,
            inertia,
            marker='o'
        )

        plt.xlabel("Number of Clusters")

        plt.ylabel("Inertia")

        plt.title("Elbow Method")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig("Elbow_Method.png", dpi=300)

        plt.show()

    # ======================================================
    # KMeans Clustering
    # ======================================================

    def kmeans(self, clusters=3):

        model = KMeans(

            n_clusters=clusters,

            random_state=42,

            n_init=10

        )

        labels = model.fit_predict(self.df)

        pca = PCA(n_components=2)

        reduced = pca.fit_transform(self.df)

        plt.figure(figsize=(10,7))

        plt.scatter(

            reduced[:,0],

            reduced[:,1],

            c=labels

        )

        plt.title("KMeans Clustering")

        plt.xlabel("PC1")

        plt.ylabel("PC2")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig("KMeans_Clusters.png", dpi=300)

        plt.show()

        score = silhouette_score(

            self.df,

            labels

        )

        print("\nSilhouette Score :", score)

        return labels

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

class PredictionModels:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    # ======================================================
    # Prepare Dataset
    # ======================================================

    def prepare_data(self, target):

        X = self.df.drop(columns=[target])

        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42

        )

        return X_train, X_test, y_train, y_test

    # ======================================================
    # Evaluate Model
    # ======================================================

    def evaluate(self, name, model, X_test, y_test):

        prediction = model.predict(X_test)

        mae = mean_absolute_error(y_test, prediction)

        mse = mean_squared_error(y_test, prediction)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, prediction)

        print("\n", "="*50)

        print(name)

        print("="*50)

        print("MAE :", mae)

        print("MSE :", mse)

        print("RMSE:", rmse)

        print("R2  :", r2)

        plt.figure(figsize=(8,6))

        plt.scatter(

            y_test,

            prediction

        )

        plt.xlabel("Actual")

        plt.ylabel("Predicted")

        plt.title(name)

        plt.grid(True)

        plt.savefig(name.replace(" ","_") + ".png", dpi=300)

        plt.show()

        return [

            name,

            mae,

            mse,

            rmse,

            r2

        ]

    # ======================================================
    # Run All Models
    # ======================================================

    def run(self, target):

        X_train, X_test, y_train, y_test = self.prepare_data(target)

        results = []

        # -----------------------
        # Linear Regression
        # -----------------------

        lr = LinearRegression()

        lr.fit(X_train, y_train)

        results.append(

            self.evaluate(

                "Linear Regression",

                lr,

                X_test,

                y_test

            )

        )

        # -----------------------
        # Decision Tree
        # -----------------------

        dt = DecisionTreeRegressor(

            random_state=42

        )

        dt.fit(X_train, y_train)

        results.append(

            self.evaluate(

                "Decision Tree",

                dt,

                X_test,

                y_test

            )

        )

        # -----------------------
        # Random Forest
        # -----------------------

        rf = RandomForestRegressor(

            n_estimators=100,

            random_state=42

        )

        rf.fit(X_train, y_train)

        results.append(

            self.evaluate(

                "Random Forest",

                rf,

                X_test,

                y_test

            )

        )

        # -----------------------
        # Support Vector Machine
        # -----------------------

        svr = SVR()

        svr.fit(X_train, y_train)

        results.append(

            self.evaluate(

                "Support Vector Regression",

                svr,

                X_test,

                y_test

            )

        )

        result_df = pd.DataFrame(

            results,

            columns=[

                "Model",

                "MAE",

                "MSE",

                "RMSE",

                "R2"

            ]

        )

        print("\n")

        print(result_df)

        result_df.to_csv(

            "Model_Comparison.csv",

            index=False

        )

        return result_df

# ==========================================================
# FINAL REPORT
# ==========================================================

class FinalReport:

    def __init__(self, dataframe, result_df):

        self.df = dataframe
        self.results = result_df

    # ======================================================
    # Feature Importance
    # ======================================================

    def feature_importance(self):

        print("\nGenerating Feature Importance...")

        X = self.df.drop(columns=["TDS"])

        y = self.df["TDS"]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(X, y)

        importance = model.feature_importances_

        feature = pd.DataFrame({

            "Feature": X.columns,

            "Importance": importance

        })

        feature = feature.sort_values(
            by="Importance",
            ascending=False
        )

        print(feature)

        feature.to_csv(
            "Feature_Importance.csv",
            index=False
        )

        plt.figure(figsize=(12,8))

        plt.barh(
            feature["Feature"],
            feature["Importance"]
        )

        plt.gca().invert_yaxis()

        plt.title(
            "Random Forest Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            "Feature_Importance.png",
            dpi=300
        )

        plt.show()

    # ======================================================
    # Model Comparison
    # ======================================================

    def comparison_chart(self):

        plt.figure(figsize=(10,6))

        plt.bar(

            self.results["Model"],

            self.results["R2"]

        )

        plt.ylabel("R2 Score")

        plt.xlabel("Machine Learning Models")

        plt.title("Model Comparison")

        plt.xticks(rotation=20)

        plt.tight_layout()

        plt.savefig(
            "Model_Comparison.png",
            dpi=300
        )

        plt.show()

    # ======================================================
    # Best Model
    # ======================================================

    def best_model(self):

        best = self.results.loc[
            self.results["R2"].idxmax()
        ]

        print("\n")

        print("="*60)

        print("BEST MODEL")

        print("="*60)

        print(best)

        return best

    # ======================================================
    # Save Optimized Dataset
    # ======================================================

    def save_dataset(self):

        self.df.to_csv(

            "Final_Optimized_Dataset.csv",

            index=False

        )

        print("\nDataset Saved Successfully.")

    # ======================================================
    # Generate Report
    # ======================================================

    def report(self):

        best = self.best_model()

        with open("ML_Report.txt","w") as file:

            file.write("="*70+"\n")

            file.write("GROUNDWATER QUALITY MACHINE LEARNING REPORT\n")

            file.write("="*70+"\n\n")

            file.write(

                f"Generated : {datetime.now()}\n\n"

            )

            file.write(

                f"Total Samples : {len(self.df)}\n"

            )

            file.write(

                f"Total Parameters : {len(self.df.columns)}\n\n"

            )

            file.write(

                "Machine Learning Models Used\n"

            )

            file.write("-----------------------------\n")

            file.write(

                "1. Linear Regression\n"

            )

            file.write(

                "2. Decision Tree\n"

            )

            file.write(

                "3. Random Forest\n"

            )

            file.write(

                "4. Support Vector Regression\n\n"

            )

            file.write(

                "Best Model\n"

            )

            file.write("-----------------------------\n")

            file.write(

                f"{best['Model']}\n"

            )

            file.write(

                f"R2 Score : {best['R2']}\n"

            )

            file.write(

                f"RMSE : {best['RMSE']}\n"

            )

            file.write(

                f"MAE : {best['MAE']}\n"

            )

        print("\nML_Report.txt Generated.")

import pandas as pd

file = "Original sample data for Manuscript (1).xlsx"

# Read Excel file
df = pd.read_excel(file, header=1)

# Remove units row and blank row
df = df.iloc[2:].reset_index(drop=True)

# Clean column names
df.columns = df.columns.str.replace("\n", "", regex=False)
df.columns = df.columns.str.strip()

# Remove unnecessary unnamed columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

# Numeric columns
numeric_columns = [
    "Depth of Borewell",
    "Latitude",
    "Longitude",
    "Ph",
    "ORP (mV)",
    "EC",
    "Temp.",
    "DO",
    "TDS",
    "F",
    "Fe",
    "Mn",
    "NO3",
    "HCO3-",
    "Ca",
    "Cl",
    "SO4-2",
    "Na",
    "K",
    "Mg",
    "Zinc (as Zn)",
    "Total Hardness (as CaCO3)",
    "COD",
    "BOD",
    "As"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.head())
# ==========================================================
# RUN DATA OPTIMIZATION
# ==========================================================

optimizer = DataOptimization(df)

optimizer.dataset_summary()

optimizer.missing_values()

optimizer.missing_value_heatmap()

optimizer.duplicate_records()

optimizer.remove_duplicates()

optimizer.fill_missing()

optimizer.correlation()

optimizer.histogram_dashboard()

optimizer.boxplot_dashboard()

scaled_df = optimizer.normalize()

# ==========================================================
# PCA + CLUSTERING
# ==========================================================

ml = MachineLearningAnalysis(scaled_df)

ml.perform_pca()

ml.explained_variance()

ml.elbow_method()

cluster_labels = ml.kmeans(3)

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

ml_models = PredictionModels(

    scaled_df

)

results = ml_models.run(

    "TDS"

)

# ==========================================================
# FINAL REPORT
# ==========================================================

report = FinalReport(

    scaled_df,

    results

)

report.feature_importance()

report.comparison_chart()

report.save_dataset()

report.report()

print("\n")

print("="*70)

print("GROUNDWATER QUALITY ML PROJECT COMPLETED")

print("="*70)


---

## Outputs

- Correlation Heatmap
- Histogram Dashboard
- Boxplot Dashboard
- PCA Projection
- KMeans Cluster Plot
- Feature Importance
- Model Comparison
- Optimized Dataset
- Machine Learning Report

---

## Project Structure

```text
Groundwater-Quality-Analysis-ML
│
├── dataset
├── output
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/Groundwater-Quality-Analysis-ML.git
```

```bash
cd Groundwater-Quality-Analysis-ML
```

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

---

## Author

Harshvardhan Nayakal

Bachelor of Mechatronics Engineering

Python | Machine Learning | IoT | Embedded Systems
