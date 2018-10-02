#！/user/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(name='TuringX',
      version='0.0.1',
      description='A machine learning structure developed by TensorFlow',
      author='Mingqi, Yuan',
      author_email='Friedrich_Gauss@foxmail.com',
      maintainer='Mingqi, Yuan',
      maintainer_email='Friedrich_Gauss@foxmail.com',
      url='http:/www.github.com/FreeAIwithGitHub/TuringX',
      packages=find_packages(),
      py_moudles = ['TFLR', 'TFLGR', 'TFR'],
      license="License",
      platforms=["any"],
      )
