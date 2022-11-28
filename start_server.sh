#!/bin/bash
pkill -9 streamlit
rm -rf /var/log/streamlit.log
nohup streamlit run "streamlit/eda.py" 1>/var/log/streamlit.log 2>&1 &
