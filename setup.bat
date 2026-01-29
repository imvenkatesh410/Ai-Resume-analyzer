@echo off
echo Installing requirements...
python -m pip install -r requirements.txt
echo Downloading Spacy Model...
python -m spacy download en_core_web_sm
echo Setup Complete.
pause
