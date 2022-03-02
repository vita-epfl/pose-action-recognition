
from setuptools import setup
import os, sys
sys.path.append(os.path.dirname(__file__))

setup(
    name='poseact',
    version=0.2,
    packages=[
        'poseact',
        'poseact.models',
        'poseact.tools',
        'poseact.utils',
        'poseact.test'
    ],
    author='Weijiang Xiong',
    author_email='weijiangxiong1998@gmail.com',
    install_requires=[
    "torch",
    "torchvision",
    "matplotlib",
    "numpy",
    "openpifpaf"
    ],
)
