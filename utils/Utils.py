"""
    @Time    : 2019/12/23 22:06
    @Author  : Runa
"""

import time
from datetime import timedelta

def get_time_different(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))