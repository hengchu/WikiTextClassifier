from setuptools import setup, find_packages

__VERSION__ = '0.0.1'

setup(name='cis700project',
      version=__VERSION__,
      packages=find_packages(),
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'prepare-data = cis700.prepare:main',
          ]
      }
)
