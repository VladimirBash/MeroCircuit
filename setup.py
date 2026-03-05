# setup.py

from setuptools import setup, find_packages

setup(
    name="merocircuit",
    version="0.1.0",
    description="Neuromorphic autoencoder with equilibrium propagation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "pytest>=7.0.0",
        "pillow>=9.0.0",
    ],
    python_requires=">=3.9",
)