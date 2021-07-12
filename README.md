An agent-based model of [Ethereum's Proof-of-Stake consensus layer](https://github.com/ethereum/eth2.0-specs).

## Beacon Runner in practice

Notebooks using the current `beaconrunner` library.

### 1. [Altair incentives](notebooks/altair_incentives.html) + [source](notebooks/altair_incentives.ipynb)

We check that Altair incentives provide the expected rewards per validator by simulating the protocol for a few epochs.

### 2. [Beacon Runner: Thunderdome](notebooks/thunderdome/thunderdome.html) + [source](notebooks/thunderdome)

We show that honest, protocol-following agents sometimes perform worse than agents who behave more prudently. This is the case when latency is bad enough that agents hedge their bets before taking action. Agents are modelled with the beacon runner validator API and simulated.

## Early notebooks

The beacon runner was built iteratively over several notebooks. Early notebooks use early iterations of the beacon runner codebase and will not function with the code contained in this package. These early notebooks however provide background to eth2 concepts and to the general approach of our simulations.

### 1. [Beacon Runner: A BeaconState cadCAD wrap](notebooks/beaconrunner/br.html) + [source](notebooks/beaconrunner)

This notebook introduces basic eth2 concepts and provides a "centralised client" implementation. We introduce the main duties of validators in eth2 phase 0: producing blocks and attesting. In this implementation, the centralised client is the only one adding blocks to the beacon chain and attesting, thus it also has perfect view of the chain. This allows us to focus on the interplay between state (the state of the beacon chain) and policies (the duties performed by the centralised client).

### 2. [Beacon Runner 2049: Liveness and inactivity leak](notebooks/beaconrunner2049/br2049.html) + [source](notebooks/beaconrunner2049)

The centralised client of the previous notebook was the only agent producing blocks and attestations. In this notebook, we introduce validators distributed over a peer-to-peer network, who exchange the blocks and attestations they produce. We assume the network is split in half, such that neither half is able to finalise the state of the beacon chain, focusing on the cryptoeconomic mechanism that allows finalisation to resume. Our implementation is still somewhat centralised, in the sense that all validators in the same half of the network have the same view of the chain (albeit a different view from the other half's).

### 3. [Beacon Runner 2050: An agent-based model of eth2](notebooks/beaconrunner2050/br2050.html) + [source](notebooks/beaconrunner2050)

We fully decentralise the model of the previous notebook by allowing each validator to have its own view of the chain. Additionally, we provide an interface to model the behaviour of validators, using a simple API. In this notebook, we implement honest validation and observe the progress of the chain.

## Starting up

You can simply run the following commands in a terminal.

```shell
git clone https://github.com/ethereum/beaconrunner.git
cd beaconrunner

###
# Optional: use a virtual environment
# Python 3.8 is required
python3.8 -m venv venv

# Unix
source venv/bin/activate

# Windows
.\venv\Scripts\activate
###

pip install -r requirements.txt
```

Once you enter the shell, you can type in

```shell
jupyter lab
```

## Docs

Some documentation is available [here](https://ethereum.github.io/beaconrunner/docs/build/html/).
