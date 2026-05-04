# Fruit Ninja YOLOv8 Bot🍎

This project uses YOLOv8 and PyTorch to detect fruit on-screen in real-time and automates mouse movements to "slice" them. It includes logic to avoid bombs and a pause/unpause toggle.

## Prerequisites

Before starting, ensure you have an NVIDIA GPU for real-time performance.

    Operating System: Windows 10/11
    Python: 3.10, 3.11, or 3.12 (Required—Python 3.13+ is currently incompatible with CUDA Torch).
    NVIDIA Drivers: Latest Game Ready Drivers

## Screenshots



https://github.com/user-attachments/assets/779bc1ab-985b-4b9e-b9ec-3163556f439d



<hr>
<img width="1657" height="927" alt="computer vision" src="https://github.com/user-attachments/assets/33cef39c-4131-45cf-a884-6b47459d6a0c" />
<hr>

## Installation Steps

***1. Clone & Prepare Folder***  
Bash  
git clone https://github.com/SpaceTurtle/Fruit-Ninja-Bot-Using-yoloV8-.git  
cd FruitNinjaBot  

***2. Create a Stable Environment***  

    You must use a compatible Python version (3.10 - 3.12).  
    PowerShell   

***3. Remove old environment if it exists***  
rm -r venv   

***4. Create new environment with Python 3.12***  
py -3.12 -m venv venv  

***5. Activate the environment***  
.\venv\Scripts\activate  

***6. Install GPU-Accelerated PyTorch***  

    Standard pip install torch often installs the CPU-only version.  
    
    PowerShell  
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  

***7. Install Dependencies***  

    PowerShell    
    pip install ultralytics opencv-python numpy pyautogui keyboard mss  

How to Run  

    Open Fruit Ninja: Set the game to Windowed Mode (1280x720 resolution recommended).
    
    PowerShell
    python fruitbot.py
    Controls:
        P: Toggle Pause/Unpause.
        Q: Quit the script safely.

## Configuration

You can adjust these variables inside fruitbot.py for better performance:

    imgsz=320: Lower this to increase FPS (speed).

    top_right, top_left: Adjust these to match your game window position.

    duration=0.15: Change the speed of the mouse "slash."

📄 License

[MIT](https://mit-license.org/)
