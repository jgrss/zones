import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


lib_name = 'zones'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Zonal statistics on raster data'
git_url = 'http://github.com/jgrss/zones.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('requirements.txt') as f:
    required_packages = f.read()

# Parse the version from the module.
# Source: https://github.com/mapbox/rasterio/blob/master/setup.py
with open('zones/version.py') as f:

    for line in f:

        if line.find("__version__") >= 0:

            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")

            continue


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'zones': ['datasets/raster/*.tif',
                      'datasets/vector/*.cpg',
                      'datasets/vector/*.dbf',
                      'datasets/vector/*.prj',
                      'datasets/vector/*.qpj',
                      'datasets/vector/*.shp',
                      'datasets/vector/*.shx',
                      'datasets/vector/*.gpkg',
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
                    version=version,
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
