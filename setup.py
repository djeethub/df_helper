from setuptools import setup

setup(
    name='df_helper',
    version='0.0.16',
    description='df_helper',
    packages=['df_helper'],
    url='https://github.com/djeethub/df_helper.git',
    install_requires=[
      'torch', 'safetensors'
    ],
)