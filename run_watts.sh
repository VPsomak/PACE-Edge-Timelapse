#!/bin/bash
python3 main.py approximation newman_watts_strogatz;
python3 main.py greedy1 newman_watts_strogatz;
python3 main.py greedy2 newman_watts_strogatz;
python3 main.py genetic1 newman_watts_strogatz;
python3 main.py genetic2 newman_watts_strogatz;
python3 main.py ilp1 newman_watts_strogatz;
python3 main.py ilp2 newman_watts_strogatz;