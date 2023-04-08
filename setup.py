from setuptools import setup, find_namespace_packages

setup(name='w281_final',
    version='0.01',
    python_requires='>=3.9',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    install_requires=[
        'imageio',
        'tensorflow',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ],
)
