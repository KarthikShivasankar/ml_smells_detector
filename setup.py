from setuptools import setup, find_packages

setup(
    name="ml_code_smell_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "astroid",
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "torch",
        "transformers",
    ],
    entry_points={
        'console_scripts': [
            'ml_smell_detector=ml_code_smell_detector.cli:main',
        ],
    },
    author="Karthik Shivashankar",
    author_email="karthik13sankar@outlook.com",

    description="A package to detect code smells in machine learning code",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KarthikShivasankar/ml_code_smell_detector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)