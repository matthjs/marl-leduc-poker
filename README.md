<br />
<p align="center">
  <h1 align="center">Solving Leduc Poker with Counterfactual Regret
Minimization</h1>

  <p align="center">
  </p>
</p>

## Installation Instructions

To set up the `marl-leduc-poker` project, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/matthjs/marl-leduc-poker
cd marl-leduc-poker
```

2. Install the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/):
```bash
sudo snap install astral-uv --classic
```

3. Install dependencies:
```bash
uv sync
```
4. To add more libraries, you can use:
```bash
uv add <package_name>
```

## Running the project

To run the project, you can use uv:
```bash
uv run python <script_name>
```

Alternatively, you can activate the virtual environment and run Python scripts directly:
```bash
source .venv/bin/activate
python <script_name>
```

See the `main.py` script for running experiments/reproducing results.
