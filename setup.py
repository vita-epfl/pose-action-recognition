
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
    ]
)
