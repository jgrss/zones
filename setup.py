import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


__version__ = '0.2.2'

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
                     'geopandas',
                     'xarray',
                     'pandas',
                     'bottleneck',
                     'six',
                     'shapely',
                     'earthpy']


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'zones': ['datasets/raster/*.tif',
                      'datasets/vector/*.cpg',
                      'datasets/vector/*.dbf',
                      'datasets/vector/*.prj',
                      'datasets/vector/*.qpj',
                      'datasets/vector/*.shp',
                      'datasets/vector/*.shx',
                      'helpers/*.so',
                      'helpers/*.cpp']}


def get_extensions():

    return [Extension('*',
                      sources=['zones/helpers/_dictionary.pyx'],
                      language='c++')]


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
                    package_data=get_package_data(),
                    packages=get_packages(),
                    ext_modules=cythonize(get_extensions()),
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
