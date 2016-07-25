from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('License.txt') as f:
    license = f.read()

setup(
    name='state_farm',
    version='0.0.1',
    description='lean data and predict loan rates',
    long_description=readme,
    author='Joshua Arnold',
    author_email='j.arnold.111@gmail.com',
    url='https://github.com/arnold-jr/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)