# Dedicated to the public domain under CC0: https://creativecommons.org/publicdomain/zero/1.0/.

# setuptools setup script.
# users should install with: `$ pip3 install cove`
# developers can make a local install with: `$ pip3 install -e .`
# upload to pypi test server with: `$ python3 setup.py sdist upload -r pypitest`
# upload to pypi prod server with: `$ python3 setup.py sdist upload`

from setuptools import setup


long_description = '''\
Cove is a code coverage tracing harness.
'''

setup(
  name='cove',
  license='CC0',
  version='0.0.0',
  author='George King',
  author_email='george.w.king@gmail.com',
  url='https://github.com/gwk/cove',
  description='Cove is a code coverage tracing harness.',
  long_description=long_description,
  install_requires=['pithy'],
  py_modules=['cove'],
  entry_points = {'console_scripts': [
    'cove=cove:main',
  ]},
  keywords=['testing'],
  classifiers=[ # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Software Development :: Testing',
  ],
)
