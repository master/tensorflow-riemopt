from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="tensorflow-riemopt",
    version="0.2.0",
    description="a library for optimization on Riemannian manifolds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Oleg Smirnov",
    author_email="oleg.smirnov@gmail.com",
    packages=find_packages(),
    install_requires=["tensorflow<2.12.0", "keras<2.12.0", "protobuf<3.20,>=3.9.2"],
    python_requires=">=3.6.0",
    url="https://github.com/master/tensorflow-riemopt",
    zip_safe=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent",
    ],
    keywords="tensorflow optimization machine learning",
)
