from setuptools import setup, find_packages
from setuptools.command.install import install
import os


setup(
    name='orthorec',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    author_email='vnikitin@anl.gov',
    url='https://github.com/xray-imaging/orthorec',
    packages=find_packages(),
    include_package_data = True,
    scripts=['bin/orthorec'],
    description='cli to run orthorec at the Advanced Photon Source',
    zip_safe=False,
)