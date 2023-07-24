import os

import regulAS

from pathlib import Path
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='regulAS',
    version=regulAS.__version__,
    packages=['regulAS', 'regulAS.core', 'regulAS.utils', 'regulAS.reports', 'regulAS.persistence'],
    package_data={'': [os.path.join('conf', '*.yaml')]},
    url='https://github.com/slipnitskaya/regulAS',
    author='Sofya Lipnitskaya',
    author_email='lipnitskaya.sofya@gmail.com',
    description='Bioinformatics Tool for the Integrative Analysis of Alternative Splicing Regulome using RNA-Seq data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    install_requires=[
        'networkx==2.5',
        'numpy==1.19.2',
        'scipy==1.7.1',
        'scikit-learn==0.24.2',
        'pandas==1.2.2',
        'matplotlib==3.1.3',
        'omegaconf==2.1.0',
        'hydra-core==1.1.0',
        'hydra-colorlog==1.1.0',
        'sqlalchemy==1.4.17',
        'msgpack==1.0.2',
        'msgpack-numpy==0.4.6.1',
        'requests==2.26.0',
        'pyensembl==1.9.1',
        'tqdm==4.46.0'
    ]
)
