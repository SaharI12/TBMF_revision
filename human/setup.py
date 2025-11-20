from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tbmf-transformer",
    version="0.1.0",
    author="Sahar I",
    author_email="your.email@example.com",
    description="Vision Transformer for Quantitative MRI Parameter Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaharI12/TBMF",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchmetrics>=0.11.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "h5py>=3.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "Pillow>=8.3.0",
        "shap>=0.41.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "notebook>=6.0.0",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "tbmf-main=main:main",
        ],
    },
)