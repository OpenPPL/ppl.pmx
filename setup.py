from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="opmx",
    version="0.1.0",
    author="OpenPPL",
    description="Open PPL Model Exchange (OPMX) - An open ecosystem for AI model exchange",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenPPL/ppl.pmx",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)