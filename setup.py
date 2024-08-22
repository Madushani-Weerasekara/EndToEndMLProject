# Building our application as a package itself.

from setuptools import find_packages, setup





setup(

    name='mlprojet',
    version='0.0.1',
    author='Madushani',
    author_email='mmweerasekara@gmail.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn']
)