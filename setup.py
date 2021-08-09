# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

from setuptools import setup


name = 'coven'

setup(
  long_description=open('readme.wu').read(),
  entry_points = {'console_scripts': [
    'coven=coven.main:main',
    'coven-dis=coven.disassemble:main',
  ]},
)
