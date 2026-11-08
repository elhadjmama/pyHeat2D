from setuptools import setup, find_packages

with open("./README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyHeat2D",
    version="0.0.1",
    author="El Hadji Mama GUENE",
    url="https://github.com/elhadjmama/pyHeat2D",
    description="A generic two-dimensional solver for the heat diffusion problem using the finite difference method. "
                "Designed to adapt to various boundary conditions encountered in physics, this solver provides a flexible "
                "framework that allows researchers to customize it for specific applications, from studying heat transfer "
                "to analyzing thermal processes in engineering systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=["numpy",
                      "matplotlib",
                      "scipy",
                      "sympy"],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Education",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
