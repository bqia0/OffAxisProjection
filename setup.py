
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OffAxisProjection",
    version="0.0.1",
    author="Brandon Qiao",
    author_email="bjqiao2@illinois.edu",
    description="Pixel Densities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bqia0/OffAxisProjection",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ),
)