import setuptools
from distutils.core import setup

import numpy as np


__version__ = '0.0.2dev'

lib_name = 'zones'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Zonal statistics on raster data'
git_url = 'http://github.com/jgrss/zones.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

required_packages = ['GDAL',
                     'numpy',
                     'tqdm',
                     'future',
                     'geopandas',
                     'pandas',
                     'bottleneck']


def get_packages():
    return setuptools.find_packages()


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=lib_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    packages=get_packages(),
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
