from os import path

from setuptools import setup

__VERSION__ = '0.3.1'

with open("README.md", "r") as f:
    long_description = f.read()

here = path.abspath(path.dirname(__file__))


setup(
    name='darkgreybox',
    version=__VERSION__,
    description='DarkGreyBox: An Open-Source Data-Driven Python Building Thermal Model'
                ' Inspired By Genetic Algorithms and Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='python model thermal machine-learning genetic-algorithm data-science',

    # The project's main homepage.
    url='https://github.com/czagoni/darkgreybox',
    download_url='https://github.com/czagoni/darkgreybox/archive/refs/tags/0.3.1.tar.gz',

    # Author details
    author='csaba zagoni',
    author_email='czagoni@greenpeace.org',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],

    # What does your project relate to?

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['darkgreybox'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'lmfit~=1.1.0',
        'pandas~=1.5.2',
        'joblib~=1.2.0'
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'statsmodels~=0.13.1',
            'numdifftools~=0.9.39',
            'scikit-learn~=1.2.0',
            'matplotlib~=3.6.3',
            'jupyter~=1.0.0',
            'notebook~=6.5.2',
            'autopep8~=2.0.1',
            'isort~=5.11.4',
        ],
        'test': [
            'flake8~=6.0.0',
            'pytest~=7.2.1',
            'pytest-mock~=3.10.0',
            'pytest-cov~=4.0.0',
        ],
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    },

    include_package_data=True
)
