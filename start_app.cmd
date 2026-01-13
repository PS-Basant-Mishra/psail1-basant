@echo off
echo Starting AI Assistant...
start http://127.0.0.1:5001
pip install -r requirements.txt
pip install faiss-cpu
python app.py
pause
