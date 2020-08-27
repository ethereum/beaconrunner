from setuptools import setup, find_packages

setup(
    name='beaconrunner',
    url='https://github.com/barnabemonnot/beaconrunner',
    author='Barnabé Monnot',
    author_email='barnabe.monnot@ethereum.org',
    packages=find_packages(include=['beaconrunner', 'beaconrunner.*']),
    install_requires=['eth2spec', 'cadCAD', 'pandas', 'milagro_bls_binding'],
    version='0.1.3',
    license='MIT',
    description='Agent-based simulation environment for eth2',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
)
