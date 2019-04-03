from setuptools import find_packages, setup

setup(
    name='librispect',
    packages=find_packages(),
    version='0.1.0',
    description='Utilities for downloading the librivox dataset and converting to spectrogram for use with keras fit_generator',
    author='MarvinT',
    license='MIT',
)
