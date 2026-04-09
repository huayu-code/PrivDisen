from setuptools import setup, find_packages

setup(
    name="privdisen",
    version="0.1.0",
    description="PrivDisen: Privacy-Preserving Label Protection via Variational "
                "Disentangled Representation in Vertical Federated Learning",
    author="Yang Tan",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.11.0",
        "tqdm>=4.64.0",
    ],
)
