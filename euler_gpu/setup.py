from setuptools import setup, find_packages

setup(
    name="euler_gpu",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["numpy", "torch"],
    author="Adam Atanas",
    author_email="adamatanas@gmail.com",
    description="Pytorch-based implementation of Euler registration using GPU acceleration.",
    license="MIT",
    url="http://github.com/flavell-lab/EulerGPU"
)

