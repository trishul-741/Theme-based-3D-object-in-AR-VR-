# Theme-based-3D-object-in-AR-VR-

This project combines multimodal AI (text + multiple images) to generate theme-based 3D models. It uses a Streamlit web app for input and provides a web-based AR viewer to place your creations in the real world using your phone.

# Core Features
Multimodal Input: Generates 3D models from a text prompt and multiple reference images.
AI Generation: Uses Shap-E for 3D mesh generation and Stable Diffusion for texturing.
Web-Based AR: Creates a self-contained AR package using Google's <model-viewer>.
Simple UI: An easy-to-use Streamlit app for generating models and getting the AR link.

# Setup and Installation
Follow these steps precisely to set up your environment.

## 1. Create Virtual Environment
You must use Python 3.10 for compatibility.

Bash
python -m venv venv

Activate the environment:
Windows: .\venv\Scripts\activate
Mac/Linux: source venv/bin/activate

## 2. Install PyTorch
You must install PyTorch with CUDA support before installing other packages. Visit the Official PyTorch Website and run the command for your system (e.g., for CUDA 11.8):
Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## 3. Install Shap-E
The shap-e library is not on PyPI. You must install it from GitHub.
Bash
git clone https://github.com/openai/shap-e.git
pip install -e ./shap-e

## 4. Install Dependencies
Install the remaining packages from your requirements file and the correct version of pyglet.
Bash
pip install -r requirements.txt
pip install "pyglet<2"

## 5. Set Up ngrok
This tool creates a secure HTTPS link, which is required for your phone's browser to access the camera for AR.
Download ngrok from the official website.
Unzip ngrok.exe into your project directory.
Sign up for a free ngrok account to get an Authtoken.

Run this command once to link your account (replace <YOUR_TOKEN>):
PowerShell
.\ngrok config add-authtoken <YOUR_TOKEN_HERE>

# How to Run the System
You will need three separate terminals running at the same time.

## Terminal 1: Run the Streamlit App
This terminal runs the main web application.
Make sure your virtual environment is active.
Run the app:
Bash
streamlit run multimodal_app.py
Open the app in your browser (e.g., http://localhost:8501).

Enter your text prompt and upload your images.
Click "Generate 3D Model" and wait for it to complete.

## Terminal 2: Run the Local Server
This terminal hosts the 3D model files.
After generation, the Streamlit app will show a path under "AR/VR Viewer". It will look like: cd D:\ML CP\cp\outputs\ar_package
Open a new terminal.
cd into that exact ar_package directory:
PowerShell
### Example:
cd D:\ML CP\cp\outputs\ar_package
Start the local server:
Bash
python -m http.server 8000

## Terminal 3: Run ngrok
This terminal creates the secure public link for your phone.
Open a third terminal.
cd into your main project directory (where ngrok.exe is):
PowerShell
cd D:\ML CP\cp
Start ngrok to tunnel to your local server:
PowerShell
.\ngrok http 8000
ngrok will give you a public HTTPS URL. It will look like this: Forwarding https://<random-string>.ngrok-free.app -> http://localhost:8000

# View in AR
On your phone, open the https://... URL from Terminal 3.
The 3D model will load in the viewer (it will spin).
Tap the AR icon in the bottom-right corner.
Point your phone at the floor to place the 3D object in your room.
