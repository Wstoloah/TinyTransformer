from setuptools import setup, find_packages

setup(
    name="tinytransformer",
    version="0.1.0",
    description="A minimal transformer implementation for learning and experimentation.",
    author="Your Name",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "torch>=2.0.0",
        "tqdm>=4.0.0",
    ],
    python_requires=">=3.8",
)
