<br />
<p align="center">
  <h1 align="center">Insert Project Name</h1>

  <p align="center">
  </p>
</p>

## About The Project

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

## Training

### Configuration

The training script uses Hydra for configuration management with organized configuration files:

[...]

### Environment Setup

Create a `.env` file in the project root to store environment variables. You can start by copying the example file:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual values:
```bash
# Example .env file
WANDB_API_KEY=your_wandb_api_key_here
# Add other environment variables as needed
```
