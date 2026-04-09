from setuptools import setup, find_packages

setup(
    name="privdisen",
    version="0.1.0",
    description="Privacy-Preserving Disentangled Representation Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
)
