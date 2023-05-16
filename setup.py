from setuptools import setup, find_packages

setup(
    name='df_helper',
    version='0.0.3',
    description='df_helper',
    url='https://github.com/djeethub/df_helper.git',
    packages=find_packages(),
    install_requires=[
      'safetensors'
    ],
)