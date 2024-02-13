from setuptools import setup, find_packages

setup(name='alfred',
      version='0.0.1',
      description='Just some boilerplate code for machine learning projects',
      url='https://github.com/julienroy13/alfred',
      author='Julien Roy',
      packages=find_packages(),
      install_requires=[
            'numpy>=1.16.3',
            'matplotlib>=3.1.2'
      ]
)
