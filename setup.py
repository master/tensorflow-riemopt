from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="tensorflow-riemopt",
    version="0.3.0",
    description="a library for optimization on Riemannian manifolds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Oleg Smirnov",
    author_email="oleg.smirnov@gmail.com",
    packages=find_packages(),
    install_requires=["tensorflow"],
    python_requires=">=3.10.0",
    url="https://github.com/master/tensorflow-riemopt",
    zip_safe=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent",
    ],
    keywords="tensorflow optimization machine learning",
)
