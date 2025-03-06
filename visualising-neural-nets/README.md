




# Visualising Neural Networks Learn

 
<hr>

## Overview 📌

*Working Progress*

- This may be my favourite one
- There is just so much to talk about, my goodness

<hr>

## Demo 📸

## Setup ⚙️

- Ensure you are in the root folder: `visualising-neural-nets/`

### Automated Setup

Run the appropriate script based on your OS

- **Linux/MacOS:**
    ```bash
    ./scripts/demo.sh
    ```

- **Windows:**
    ```bash
    ./scripts/demo.bat
    ```
This script will:
1. Create a virtual environment
2. Install dependencies
3. Run the selected script
4. Finally, clean up

### Manual

Alternatively, you could create & activate a virtual environment yourself (this would offer more flexibility of course).

- Create Virtual Environment
    ```bash
    python -m venv env
    ```
- Activate Virtual Environment
    ```bash
    source env/bin/activate
    ```
- Upgrade pip within the Virtual Environment
    ```bash
    pip install --upgrade pip
    ```
- Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

- Run function approximator in 2D
    ```bash
    python main.py
    ```

- Run function approximator in 3D
    ```bash
    python 3d-main.py
    ```

- See the output of a neural network
    ```bash
    python ml-img.py
    ```

- Deactivate Virtual Environment
    ```bash
    deactivate
    ```

