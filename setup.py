from setuptools import setup, find_packages

setup(
    name="discourse-classification",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.7",
        "pyyaml>=6.0",
        "pytorch-crf>=0.7.2",
    ],
)
