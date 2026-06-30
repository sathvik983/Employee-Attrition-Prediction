@echo off
echo ===================================================
echo   Employee Attrition Prediction System (TalentGuard)
echo ===================================================
echo.
echo Step 1: Installing python packages...
pip install -r requirements.txt
echo.
echo Step 2: Running Data Preparation...
python data_loader.py
echo.
echo Step 3: Training Machine Learning Models...
python model_trainer.py
echo.
echo Step 4: Launching Dashboard Application...
python -m streamlit run app.py --server.port 8505
pause
