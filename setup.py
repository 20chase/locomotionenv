import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

env_assets = package_files('locomotionenv/envs/assets')


setup(
    name='locomotionenv',
    version='1.0.0',
    packages=find_packages(),
    description='Robot MuJoCo environments with Gym API.',
    long_description=read('README.md'),
    install_requires=[
        'click', 'gym',  'numpy',
    ],
    package_data={
        'locomotionenv': env_assets,
    },
)
