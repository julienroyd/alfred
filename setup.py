from setuptools import setup, find_packages

setup(name='alfred',
      version='0.0.1',
      description='Just some boilerplate code for machine learning projects',
      url='https://github.com/julienroy13/alfred',
      author='Julien Roy',
      packages=find_packages(),
      install_requires=[
            'tqdm>=4.40.1',
            'seaborn>=0.9.0'
      ]
)
