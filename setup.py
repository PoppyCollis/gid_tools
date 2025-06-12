from setuptools import setup, find_packages

setup(
    name="GIDTools",
    version="0.1.0",
    author="Poppy Collis",
    author_email="pzc20@sussex.ac.uk",
    description="Online generative inverse design of tools for robotic manipulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PoppyCollis/gid_tools",
    license="MIT",
    packages=find_packages(include=["gid_tools", "gid_tools.*"]),
    python_requires=">=3.7",
    install_requires=[
        # e.g. "numpy>=1.20", "torch>=1.9", …
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

