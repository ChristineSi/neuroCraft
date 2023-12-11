from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='neurocraft',
      description="NeuroCraft Model (api_pred)",
      license="MIT",
      author="neuroCraft Team - Le Wagon",
      #url="https://github.com/ChristineSi/neuroCraft",
      install_requires=requirements,
      packages=find_packages())
