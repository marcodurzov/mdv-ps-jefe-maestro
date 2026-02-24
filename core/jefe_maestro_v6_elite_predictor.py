import pandas as pd
import numpy as np
import psutil
import requests
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import gspread
from google.oauth2.service_account import Credentials

import os
import sys
import math
import random
import datetime

def main():
    print("Iniciando Jefe Maestro Elite Predictor...")
    
    # Si tu modelo ya tiene una función principal diferente,
    # reemplaza "run()" por el nombre real.
    
    try:
        run()
    except NameError:
        print("No existe función run(). Ajustar nombre de función principal.")
