# Beacon Runner

An agent-based model of [eth2](https://github.com/ethereum/eth2.0-specs).

## Starting up

You can simply run the following commands in a terminal, assuming `pipenv` is installed on your machine.

```
git clone https://github.com/barnabemonnot/beaconrunner.git
cd beaconrunner
git clone https://github.com/danlessa/cadCAD.git
cd cadCAD
git checkout tweaks
cd ..
pipenv install
pipenv shell
```

Once you enter the shell, you can type in

```
jupyter lab
```

## Docs

Some documentation is available [here](https://barnabemonnot.com/beaconrunner/build/html/).
