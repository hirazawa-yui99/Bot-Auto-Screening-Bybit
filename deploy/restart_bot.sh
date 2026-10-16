#!/bin/bash
set -e
cd /home/ubuntu/bybit_quant_v9
pkill -f "python main.py" || true
sleep 2
nohup ./venv/bin/python main.py > bot_v9.log 2>&1 &
echo "Bot restarted (PID $!)"