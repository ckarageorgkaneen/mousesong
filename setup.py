import setuptools

setuptools.setup(
    name="mousesong",
    version="0.1",
    author="Christos Karageorgiou Kaneen",
    author_email="ckarageorgkaneen@gmail.com",
    url="https://github.com/ckarageorgkaneen/mousesong",
    description="Modeling singing mouse behavior.",
    packages=setuptools.find_packages(),
    python_requires="==3.7.12",
    install_requires=[
        "sleap[pypi]==1.4.1a2",
        "opencv-python",
        "ffmpeg",
        "pyyaml",
        "scipy",
    ],
)
