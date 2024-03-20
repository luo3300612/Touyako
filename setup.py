#! /usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import setuptools

setup(
name = 'touyako',  # 包的名字
author = 'luo3300612',  # 作者
version = '0.1.0',  # 版本号
license = 'MIT',
description = '',  # 描述
long_description = '''''',
author_email = 'lyricpoem1997@gmail.com',  # 你的邮箱**
url = 'https://github.com/tansor',  # 可以写github上的地址，或者其他地址
# 依赖包
install_requires = [
                       # "prettytable", "lmdb", "opencv-python", "numpy",
        # "tqdm", "seaborn", "pandas", "matplotlib", "pillow", "swifter"
    ],
    classifiers=[
        # 'Development Status :: 4 - Beta',
        # 'Operating System :: Microsoft'  # 你的操作系统  OS Independent      Microsoft
        'Intended Audience :: Developers',
        # 'License :: OSI Approved :: MIT License',
        # 'License :: OSI Approved :: BSD License',  # BSD认证
        'Programming Language :: Python',  # 支持的语言
        'Programming Language :: Python :: 3',  # python版本 。。。
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries'
    ],
    zip_safe=True,
)