***
# 🔧 Installation Instructions
### 🖥️ Prerequisites

* Python 3.12+
* Git 
***


### ⚡ Using uv (recommended)
#### 🐧 Linux / 🍎 macOS
```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Sync dependencies from uv.lock
uv sync
```

#### 🪟 Windows (PowerShell)
```
# Install uv (if not already installed)
irm https://astral.sh/uv/install.ps1 | iex

# Create and activate a virtual environment
uv venv .venv
.venv\Scripts\Activate.ps1

# Sync dependencies
uv sync
```
***

### 📦 Using pip
#### 🐧 Linux / 🍎 macOS

First Create and activate a virtual environment (using `venv`)
```
python3 -m venv .venv
source .venv/bin/activate
```
You can of course use Anaconda or your favourite python virtual environment. Then install the dependencies.

*Note*: We have commented out packages that support GPU; feel free to put them back if you have a CUDA-enabled GPU on your machine. You shouldn't worry about it when using Google Colab though.

```
pip install -r requirements.txt
```

#### 🪟 Windows (CMD)
```
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

#### 🪟 Windows (PowerShell)
```
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```
