from setuptools import setup, find_packages

setup(
    name="systematic-equity-alpha",
    version="0.1.0",
    description="A production ML system for systematic trading with 1999-2025 Bloomberg risk factors",
    author="Donaire Aengus Martin Gilboy",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "optuna>=3.4",
        "shap>=0.44",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.15",
        "pyyaml>=6.0",
        "joblib>=1.3",
    ],
)