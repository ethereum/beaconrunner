from setuptools import setup

setup(
    name='Beacon Runner',
    url='https://github.com/barnabemonnot/beaconrunner',
    author='Barnabé Monnot',
    author_email='barnabe.monnot@ethereum.org',
    packages=['beaconrunner'],
    install_requires=['eth2spec', 'cadCAD'],
    version='0.1',
    license='MIT',
    description='Agent-based simulation environment for eth2',
    long_description=open('README.md').read(),
)
