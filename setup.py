from setuptools import setup

with open('README.md', 'r') as fp:
    long_description = fp.read()

setup(
    name='robustgp',
    version='2.0',
    description='Robust Gaussian process regression based on iterative trimming.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/syrte/robustgp/',
    keywords=['Gaussian process', 'regression', 'robust statistics', 'outlier detection'],
    author='Zhaozhou Li',
    author_email='styr.py@gmail.com',
    packages=['robustgp'],
    install_requires=['gpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
