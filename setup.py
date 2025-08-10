from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="y5gfunc",
    author="yuygfgg",
    author_email="me@yuygfgg.xyz",
    description="Yuygfgg's collection for vapoursynth video filtering and encoding stuff.",
    url="https://github.com/yuygfgg/y5gfunc",
    provides=["y5gfunc"],
    license="GPL-3.0-or-later",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
)
