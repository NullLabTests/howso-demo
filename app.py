import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

try:
    from howso.engine import Trainee
    from howso.utilities import infer_feature_attributes
    HOWSO_AVAILABLE = True
except ImportError:
    HOWSO_AVAILABLE = False

st.set_page_config(
    page_title="Howso AI Demo",
    page_icon="🤔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight-blue {
        color: #667eea;
        font-weight: 600;
    }
    .howso-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stApp { max-width: 100%; }
</style>
""", unsafe_allow_html=True)

FEATURE_NAMES = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
TARGET = 'mpg'
ORIGIN_MAP = {1: 'USA', 2: 'Europe', 3: 'Japan'}

EMBEDDED_CSV = """mpg,cylinders,displacement,horsepower,weight,acceleration,model_year,origin
18.0,8,307.0,130.0,3504,12.0,70,1
15.0,8,350.0,165.0,3693,11.5,70,1
18.0,8,318.0,150.0,3436,11.0,70,1
16.0,8,304.0,150.0,3433,12.0,70,1
17.0,8,302.0,140.0,3449,10.5,70,1
15.0,8,429.0,198.0,4341,10.0,70,1
14.0,8,454.0,220.0,4354,9.0,70,1
14.0,8,440.0,215.0,4312,8.5,70,1
14.0,8,455.0,225.0,4425,10.0,70,1
15.0,8,390.0,190.0,3850,8.5,70,1
15.0,8,383.0,170.0,3563,10.0,70,1
14.0,8,340.0,160.0,3609,8.0,70,1
15.0,8,400.0,150.0,3761,9.5,70,1
14.0,8,455.0,225.0,3086,10.0,70,1
24.0,4,113.0,95.0,2372,15.0,70,3
22.0,6,198.0,95.0,2833,15.5,70,1
18.0,6,199.0,97.0,2774,15.5,70,1
21.0,6,200.0,85.0,2587,16.0,70,1
27.0,4,97.0,88.0,2130,14.5,70,3
26.0,4,97.0,46.0,1835,20.5,70,2
25.0,4,110.0,87.0,2672,17.5,70,2
24.0,4,107.0,90.0,2430,14.5,70,2
17.0,8,304.0,145.0,3425,13.0,70,1
16.0,8,318.0,150.0,3633,11.5,70,1
21.0,6,250.0,106.0,3324,13.5,71,1
26.0,4,121.0,78.0,2752,15.5,70,2
28.0,4,97.0,57.0,2188,17.5,70,2
26.0,4,108.0,79.0,2528,15.5,70,1
27.0,4,97.0,75.0,2171,16.0,70,1
26.0,4,97.0,72.0,2265,16.0,70,1
25.0,4,110.0,78.0,2696,17.5,71,2
28.0,4,97.0,61.0,2246,17.0,71,2
28.0,4,97.0,61.0,2216,17.0,71,2
27.0,4,97.0,62.0,2194,16.5,71,2
28.0,4,97.0,63.0,2191,16.5,71,2
27.0,4,97.0,65.0,2228,16.5,71,2
23.0,4,120.0,69.0,2451,17.5,71,2
25.0,4,97.0,67.0,2290,17.0,71,2
23.0,4,110.0,68.0,2607,16.5,71,2
21.0,6,200.0,85.0,2875,15.5,71,1
21.0,6,200.0,85.0,2875,15.5,71,1
30.0,4,97.0,48.0,1985,20.0,71,2
26.0,4,97.0,65.0,2385,16.0,71,2
14.0,8,472.0,205.0,4756,10.0,71,1
15.0,8,455.0,210.0,4500,9.0,71,1
13.0,8,455.0,215.0,4733,9.0,71,1
13.0,8,350.0,145.0,4080,12.0,71,1
14.0,8,350.0,150.0,4215,11.5,71,1
13.0,8,400.0,148.0,4665,11.0,71,1
13.0,8,351.0,148.0,4365,11.0,71,1
15.0,8,350.0,150.0,4077,12.5,71,1
12.0,8,383.0,180.0,4845,11.0,71,1
13.0,8,400.0,170.0,4746,11.5,71,1
14.0,8,455.0,225.0,4425,11.0,71,1
13.0,8,350.0,155.0,4505,11.5,71,1
13.0,8,351.0,152.0,4289,10.5,71,1
17.0,6,250.0,110.0,3520,12.0,71,1
15.0,6,250.0,107.0,3599,13.0,71,1
13.0,8,400.0,170.0,4699,11.0,71,1
13.0,8,400.0,170.0,4746,11.5,71,1
14.0,8,351.0,158.0,4464,11.5,71,1
17.0,6,232.0,100.0,3355,14.0,72,1
14.0,8,304.0,140.0,3436,11.5,72,1
15.0,8,350.0,170.0,4164,11.0,72,1
17.0,8,302.0,137.0,4002,12.5,72,1
14.0,8,351.0,148.0,4543,11.5,72,1
14.0,8,400.0,163.0,4498,12.0,72,1
14.0,8,429.0,198.0,4495,11.5,72,1
16.0,6,231.0,115.0,3035,13.5,72,1
19.0,6,250.0,105.0,3525,14.5,72,1
13.0,8,429.0,198.0,4629,11.0,72,1
13.0,8,400.0,163.0,4590,12.0,72,1
13.0,8,400.0,163.0,4615,12.0,72,1
14.0,8,455.0,225.0,4499,10.5,72,1
19.0,6,232.0,100.0,3336,14.0,72,1
26.0,4,98.0,79.0,2489,15.0,72,2
24.0,4,98.0,85.0,2425,15.0,72,2
15.0,8,350.0,165.0,4274,11.0,72,1
17.0,8,260.0,125.0,3530,12.0,72,1
23.0,4,120.0,97.0,2489,15.0,72,2
18.0,6,232.0,112.0,2901,14.5,72,1
19.0,4,97.0,78.0,2290,17.0,72,3
17.0,6,258.0,110.0,3242,14.5,72,1
18.0,4,97.0,72.0,2395,14.5,72,3
16.0,8,318.0,150.0,4615,13.5,72,1
17.0,8,302.0,130.0,4021,13.5,72,1
18.0,8,304.0,140.0,3554,12.5,72,1
16.0,8,351.0,145.0,3981,12.5,72,1
18.0,6,225.0,100.0,4055,16.0,72,1
20.0,6,250.0,100.0,3786,16.0,72,1
17.0,6,232.0,105.0,3355,14.5,72,1
15.0,8,350.0,165.0,4122,12.0,72,1
18.0,8,400.0,175.0,4516,12.0,72,1
14.0,8,351.0,148.0,4615,11.5,72,1
17.0,6,231.0,110.0,3410,15.0,73,1
15.0,8,350.0,160.0,4304,12.5,73,1
13.0,8,429.0,192.0,4603,11.0,73,1
13.0,8,455.0,210.0,4633,11.0,73,1
14.0,8,351.0,159.0,4485,13.0,73,1
14.0,8,360.0,160.0,4456,12.0,73,1
13.0,8,383.0,180.0,4815,12.5,73,1
17.0,6,258.0,112.0,3735,14.5,73,1
17.0,4,110.0,86.0,2639,16.0,73,1
19.0,6,225.0,100.0,3675,15.5,73,1
20.0,4,113.0,60.0,2390,18.0,73,2
18.0,6,228.0,95.0,3785,16.0,73,1
17.0,6,200.0,85.0,3065,16.5,73,1
19.0,6,232.0,100.0,3455,15.0,73,1
18.0,6,250.0,100.0,3515,15.5,73,1
13.0,8,350.0,150.0,4300,12.0,73,1
19.0,4,97.0,75.0,2185,16.0,72,2
19.0,6,232.0,100.0,2945,15.5,73,1
18.0,4,110.0,73.0,2220,17.0,73,2
16.0,4,105.0,71.0,2685,16.5,73,2
25.0,4,97.0,61.0,2250,17.5,73,2
18.0,4,105.0,70.0,2395,16.5,73,2
20.0,4,105.0,69.0,2403,17.0,73,2
20.0,4,105.0,65.0,2415,16.5,73,2
20.0,4,120.0,87.0,2732,16.5,73,2
19.0,4,108.0,77.0,2337,16.0,73,2
20.0,4,105.0,70.0,2365,16.5,73,2
19.0,4,108.0,77.0,2337,16.0,73,2
20.0,6,200.0,85.0,3039,16.0,73,1
19.0,6,232.0,100.0,3455,15.0,73,1
18.0,6,225.0,105.0,3825,16.0,73,1
17.0,6,250.0,100.0,3785,16.0,73,1
18.0,6,232.0,100.0,3425,16.0,73,1
18.0,6,232.0,100.0,3465,16.0,73,1
16.0,6,250.0,105.0,3975,16.0,73,1
17.0,6,250.0,100.0,3630,16.5,73,1
18.0,6,250.0,88.0,3560,16.5,73,1
15.0,6,250.0,100.0,3852,16.0,73,1
17.0,6,250.0,105.0,3815,16.5,73,1
17.0,6,250.0,100.0,3656,16.5,73,1
21.0,4,112.0,90.0,2645,15.0,73,1
16.0,8,304.0,150.0,3986,11.0,73,1
26.0,4,120.0,68.0,2265,18.5,73,2
18.0,6,232.0,100.0,3425,16.0,73,1
15.0,8,350.0,150.0,4459,13.5,73,1
17.0,8,318.0,145.0,4185,13.5,73,1
16.0,8,302.0,130.0,4152,13.5,73,1
16.0,8,304.0,135.0,4125,13.5,73,1
18.0,8,350.0,155.0,4365,13.0,73,1
15.0,8,351.0,142.0,4459,13.0,73,1
16.0,8,302.0,135.0,4254,13.0,73,1
17.0,8,409.0,157.0,4715,12.0,73,1
16.0,8,302.0,135.0,4297,13.5,73,1
21.0,6,250.0,100.0,3457,16.0,73,1
24.0,4,120.0,88.0,2945,16.0,73,1
21.0,4,120.0,88.0,2957,16.5,73,1
20.0,4,108.0,75.0,2246,17.0,73,3
25.0,4,108.0,76.0,2335,16.5,73,2
21.0,4,108.0,73.0,2275,16.5,73,3
22.0,4,108.0,74.0,2408,17.0,73,3
21.0,4,108.0,75.0,2345,16.5,73,3
20.0,4,108.0,75.0,2308,16.5,73,3
19.0,4,108.0,71.0,2290,17.0,73,3
21.0,4,108.0,73.0,2315,16.5,73,3
21.0,4,108.0,74.0,2345,17.0,73,3
23.0,4,97.0,65.0,2319,15.5,73,2
20.0,4,105.0,70.0,2451,17.0,73,2
25.0,4,97.0,65.0,2360,16.0,73,2
20.0,4,120.0,78.0,2855,16.0,73,2
19.0,4,105.0,67.0,2407,16.5,73,2
19.0,4,105.0,70.0,2410,16.5,73,2
18.0,4,105.0,69.0,2475,16.5,73,2
14.0,8,350.0,165.0,4209,12.0,71,1
15.0,8,350.0,155.0,4363,12.0,71,1
16.0,8,302.0,130.0,4295,13.5,73,1
18.0,8,318.0,145.0,4190,13.0,73,1
15.0,8,350.0,145.0,4442,13.5,73,1
15.0,8,304.0,140.0,4382,13.0,73,1
15.0,8,350.0,140.0,5102,12.5,73,1
16.0,8,302.0,130.0,4428,13.0,73,1
15.0,8,304.0,135.0,4305,13.5,73,1
19.0,8,335.0,145.0,4675,14.0,73,1
19.0,8,318.0,145.0,4495,14.0,73,1
17.0,8,400.0,145.0,4630,14.0,73,1
18.0,8,302.0,130.0,4380,14.0,73,1
15.0,8,304.0,130.0,4300,13.0,73,1
15.0,8,350.0,140.0,4808,13.5,73,1
16.0,8,304.0,130.0,4436,13.5,73,1
16.0,8,400.0,145.0,4620,14.0,73,1
16.0,8,307.0,145.0,4354,13.5,73,1
14.0,8,350.0,160.0,4615,14.0,73,1
16.0,8,318.0,150.0,4490,14.0,73,1
17.0,8,400.0,155.0,4710,14.0,73,1
16.0,8,350.0,165.0,4627,11.5,71,1
15.5,8,351.0,145.0,4100,13.0,73,1
15.0,8,351.0,148.0,4425,13.5,73,1
16.0,8,350.0,150.0,4515,12.5,73,1
16.0,8,351.0,148.0,4625,13.0,73,1
15.0,8,400.0,167.0,4901,12.5,73,1
15.0,8,400.0,150.0,4665,13.0,73,1
17.0,8,302.0,129.0,4105,14.0,73,1
14.0,8,302.0,135.0,4300,13.5,73,1
16.0,8,351.0,153.0,4456,13.5,73,1
15.0,8,304.0,130.0,4475,13.0,73,1
16.0,8,360.0,145.0,4615,13.0,73,1
15.0,8,351.0,138.0,4464,13.5,73,1
16.0,8,302.0,135.0,4385,13.5,73,1
14.0,8,350.0,155.0,4456,12.0,73,1
17.0,6,232.0,105.0,3045,14.5,74,1
19.0,6,250.0,105.0,3605,15.0,74,1
18.0,6,250.0,105.0,3645,14.5,74,1
19.0,6,250.0,100.0,3460,15.5,74,1
15.0,8,350.0,150.0,3897,13.0,74,1
14.0,8,351.0,148.0,4657,13.5,74,1
17.0,6,231.0,110.0,3475,14.5,74,1
16.0,6,250.0,105.0,3645,14.5,74,1
15.0,6,250.0,110.0,3685,14.0,74,1
17.0,6,258.0,115.0,3430,15.0,74,1
15.0,8,302.0,140.0,4470,13.0,74,1
15.0,8,351.0,148.0,4457,12.5,74,1
20.0,6,225.0,100.0,3615,16.5,74,1
18.0,6,250.0,100.0,3460,16.0,74,1
22.0,4,122.0,80.0,2460,16.0,74,2
18.0,4,97.0,62.0,2300,17.5,74,3
16.0,4,97.0,68.0,2403,16.5,74,3
18.0,4,105.0,67.0,2370,17.0,74,3
21.0,4,97.0,66.0,2395,16.0,74,3
22.0,4,110.0,67.0,2395,16.0,74,3
17.0,4,97.0,67.0,2430,16.5,74,3
21.0,4,97.0,68.0,2480,16.0,74,1
18.0,4,97.0,67.0,2425,16.5,74,3
19.0,4,105.0,63.0,2345,17.0,74,3
21.0,4,97.0,66.0,2340,16.5,74,3
19.0,4,105.0,63.0,2385,16.5,74,3
21.0,4,97.0,63.0,2300,17.5,74,3
20.0,4,105.0,63.0,2455,17.0,74,3
15.0,8,350.0,145.0,4310,14.0,74,1
15.0,8,350.0,145.0,4360,14.0,74,1
18.0,6,231.0,105.0,3120,15.5,74,1
19.0,6,250.0,100.0,3645,16.0,74,1
19.0,6,250.0,105.0,3625,16.0,74,1
18.0,6,250.0,100.0,3680,16.0,74,1
16.0,6,250.0,105.0,3760,15.5,74,1
16.0,6,232.0,100.0,3305,16.0,74,1
20.0,6,250.0,100.0,3505,16.5,74,1
16.0,6,232.0,95.0,3460,17.0,74,1
20.0,6,250.0,105.0,3725,16.0,74,1
19.0,6,250.0,95.0,3535,17.0,74,1
21.0,6,200.0,85.0,3000,17.0,74,1
18.0,6,232.0,100.0,3405,16.5,74,1
20.0,6,250.0,105.0,3515,16.0,74,1
18.0,6,258.0,110.0,3420,15.5,74,1
21.0,6,225.0,95.0,3465,16.5,74,1
21.0,6,225.0,95.0,3480,16.5,74,1
21.0,6,155.0,74.0,2670,16.0,75,1
21.0,4,97.0,69.0,2484,16.5,75,2
21.0,4,98.0,70.0,2385,16.5,75,3
21.0,6,200.0,88.0,3060,17.0,75,1
22.0,4,108.0,76.0,2525,16.0,75,3
23.0,4,108.0,74.0,2489,16.0,75,3
23.0,4,108.0,74.0,2542,16.0,75,3
23.0,4,108.0,72.0,2565,16.0,75,3
23.0,4,108.0,74.0,2470,17.0,75,3
24.0,4,97.0,61.0,2293,17.0,75,2
24.0,4,97.0,65.0,2480,17.0,75,2
23.0,4,97.0,65.0,2408,17.5,75,2
21.0,4,97.0,67.0,2408,16.5,75,3
23.0,4,97.0,64.0,2420,17.5,75,3
19.0,6,232.0,105.0,3490,14.5,75,1
19.0,4,112.0,90.0,2865,16.5,75,1
19.0,6,232.0,95.0,3420,16.0,75,1
20.0,6,232.0,100.0,3410,16.0,75,1
18.0,6,250.0,100.0,3680,15.0,75,1
19.0,6,250.0,95.0,3675,16.0,75,1
19.0,6,232.0,100.0,3310,16.5,75,1
21.0,6,250.0,105.0,3700,17.0,75,1
18.0,6,258.0,110.0,3640,15.0,75,1
19.0,6,250.0,100.0,3565,16.0,75,1
18.0,6,232.0,95.0,3455,16.0,75,1
16.0,8,318.0,145.0,4035,13.0,75,1
16.0,8,304.0,140.0,4185,13.0,75,1
13.0,8,400.0,150.0,4760,12.5,75,1
14.0,8,351.0,148.0,4655,13.0,75,1
18.0,6,231.0,110.0,3415,15.5,75,1
16.0,8,350.0,140.0,4255,14.0,75,1
17.0,8,302.0,130.0,4175,14.0,75,1
16.0,8,400.0,150.0,4495,13.0,75,1
15.0,8,350.0,145.0,4390,13.5,75,1
16.0,8,318.0,150.0,4710,13.5,75,1
16.0,8,304.0,135.0,4310,14.0,75,1
16.0,8,350.0,145.0,4420,14.0,75,1
17.0,8,302.0,130.0,4320,14.5,75,1
16.0,8,304.0,130.0,4330,14.0,75,1
21.0,6,200.0,85.0,3050,16.0,75,1
18.0,6,232.0,100.0,3380,15.5,75,1
17.0,6,200.0,80.0,3070,16.0,75,1
21.0,6,200.0,84.0,3185,16.5,75,1
20.0,6,231.0,105.0,3430,16.0,75,1
20.0,4,107.0,75.0,2295,16.5,75,3
15.0,8,302.0,130.0,4320,13.5,75,1
20.0,6,200.0,88.0,3160,16.5,75,1
20.0,4,108.0,72.0,2460,17.0,75,3
30.0,4,97.0,62.0,2164,16.5,76,2
20.0,4,97.0,66.0,2341,18.0,76,2
21.0,4,97.0,63.0,2205,17.5,76,2
19.0,4,97.0,68.0,2385,17.5,76,2
[Truncated 1198 rows...]"""

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(StringIO(EMBEDDED_CSV))
        df['origin'] = df['origin'].astype(int)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_howso_full(_df):
    if not HOWSO_AVAILABLE:
        return None
    df = _df.copy()
    try:
        features = infer_feature_attributes(df)
        features[TARGET]['type'] = 'continuous'
        features['origin']['type'] = 'nominal'
        trainee = Trainee(features=features)
        trainee.train(df)
        action_features = [TARGET]
        context_features = FEATURE_NAMES
        trainee.analyze(context_features=context_features, action_features=action_features)
        return trainee
    except Exception as e:
        st.error(f"Howso training failed: {e}")
        return None

@st.cache_resource
def train_howso_targetless(_df):
    if not HOWSO_AVAILABLE:
        return None
    df = _df.copy()
    try:
        features = infer_feature_attributes(df)
        features[TARGET]['type'] = 'continuous'
        features['origin']['type'] = 'nominal'
        trainee = Trainee(features=features)
        trainee.train(df)
        trainee.analyze()
        return trainee
    except Exception as e:
        st.error(f"Howso targetless training failed: {e}")
        return None

@st.cache_resource
def train_rf_model(_df):
    df = _df.copy()
    X = df[FEATURE_NAMES]
    y = df[TARGET]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

@st.cache_data
def run_small_data_comparison(_df, _train_sizes_tuple):
    df = _df.copy()
    train_sizes = list(_train_sizes_tuple)
    results = []
    for size in train_sizes:
        if size >= len(df):
            continue
        train = df.sample(n=size, random_state=42)
        test = df.drop(train.index)
        X_train, X_test = train[FEATURE_NAMES], test[FEATURE_NAMES]
        y_train, y_test = train[TARGET], test[TARGET]

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)

        howso_r2 = None
        howso_mae = None
        if HOWSO_AVAILABLE:
            try:
                features = infer_feature_attributes(train)
                features[TARGET]['type'] = 'continuous'
                features['origin']['type'] = 'nominal'
                t = Trainee(features=features)
                t.train(train)
                t.analyze(context_features=FEATURE_NAMES, action_features=[TARGET])
                result = t.react(
                    contexts=X_test,
                    context_features=FEATURE_NAMES,
                    action_features=[TARGET],
                )
                preds = result['action'][TARGET].values
                howso_r2 = r2_score(y_test, preds)
                howso_mae = mean_absolute_error(y_test, preds)
            except Exception:
                pass
        results.append({
            'train_size': size,
            'howso_r2': howso_r2,
            'howso_mae': howso_mae,
            'rf_r2': rf_r2,
            'rf_mae': rf_mae,
            'n_test': len(test),
        })
    return pd.DataFrame(results)

st.sidebar.markdown("""
<div style="text-align:center;margin-bottom:1rem;">
    <span style="font-size:2rem;">🤔</span>
    <h2 style="margin:0;color:#1a1a2e;">Howso AI</h2>
    <p style="font-size:0.8rem;color:#6c757d;">Explainable AI Demo</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "🎯 Predict & Explain",
    "📊 Small Data Showdown",
    "🔄 Synthetic Data Quality",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem;color:#6c757d;">
    Built with ❤️ using Howso Engine<br>
    Dataset: Auto MPG (UCI)
</div>
""", unsafe_allow_html=True)

if not HOWSO_AVAILABLE:
    st.sidebar.warning("⚠️ Howso Engine not installed. Some features require `pip install howso-engine`.")

df = load_data()
if df.empty:
    st.error("Could not load dataset. Please check the deployment.")
    st.stop()

if page == "🎯 Predict & Explain":
    st.markdown("<h1 style='margin-bottom:0;'>🎯 Predict & Explain</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Predict MPG with full attribution — see <span class='highlight-blue'>why</span> every prediction is made.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)
        cyl = st.slider("Cylinders", 3, 8, 6)
        disp = st.slider("Displacement (cu in)", 68, 455, 200)
        hp = st.slider("Horsepower", 46, 230, 120)
        weight = st.slider("Weight (lbs)", 1613, 5140, 3000)
        accel = st.slider("Acceleration (0-60 mph)", 8.0, 24.8, 15.0)
        year = st.slider("Model Year", 70, 82, 76)
        origin_label = st.selectbox("Origin", ["USA", "Europe", "Japan"])
        origin = {v: k for k, v in ORIGIN_MAP.items()}[origin_label]
        predict_btn = st.button("🚀 Predict MPG", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        'cylinders': cyl, 'displacement': disp, 'horsepower': hp,
        'weight': weight, 'acceleration': accel, 'model_year': year, 'origin': origin,
    }])

    with col2:
        if not HOWSO_AVAILABLE:
            st.info("💡 Install Howso Engine to see explainable predictions: `pip install howso-engine`")
            rf_model = train_rf_model(df)
            rf_pred = rf_model.predict(input_df[FEATURE_NAMES])[0]
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-value">{rf_pred:.1f}</div>
                <div class="prediction-label">Predicted MPG (Random Forest)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📋 Feature Values</div>", unsafe_allow_html=True)
            st.dataframe(input_df, hide_index=True, use_container_width=True)
        else:
            trainee = train_howso_full(df)
            if trainee is None:
                st.error("Howso model failed to train. Check the Streamlit Cloud logs for details.")
            else:
                if predict_btn:
                    with st.spinner("Generating prediction with full attribution..."):
                        try:
                            result = trainee.react(
                                contexts=input_df,
                                context_features=FEATURE_NAMES,
                                action_features=[TARGET],
                                details={'influential_cases': True},
                            )
                            pred_value = result['action'][TARGET].iloc[0]
                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-value">{pred_value:.1f}</div>
                                <div class="prediction-label">Predicted MPG with Howso <span class="howso-badge">Attributed</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                            if 'details' in result:
                                details = result['details']
                                st.markdown("<div class='section-header'>🔍 Influential Training Cases</div>", unsafe_allow_html=True)
                                infl_cases = None
                                if isinstance(details, dict):
                                    for key in ['influential_cases', 'influentialCases', 'influential cases']:
                                        if key in details:
                                            infl_cases = details[key]
                                            break
                                if infl_cases is not None:
                                    try:
                                        infl_df = pd.DataFrame(infl_cases)
                                        if not infl_df.empty:
                                            st.dataframe(infl_df.head(10), hide_index=True, use_container_width=True)
                                            st.caption(f"Top {min(10, len(infl_df))} most influential training cases")
                                    except Exception:
                                        st.write(infl_cases)
                                else:
                                    st.caption("Influential cases data returned. Expand below to view:")
                                    with st.expander("Raw details"):
                                        st.write(details)
                        except Exception as e:
                            st.error(f"Prediction with details failed: {e}")
                            result = trainee.react(
                                contexts=input_df,
                                context_features=FEATURE_NAMES,
                                action_features=[TARGET],
                            )
                            pred_value = result['action'][TARGET].iloc[0]
                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-value">{pred_value:.1f}</div>
                                <div class="prediction-label">Predicted MPG (Howso)</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("👈 Adjust the inputs and click **Predict MPG** to see an explainable prediction with attribution.")

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Training Data Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", len(df))
    with c2:
        st.metric("Features", len(FEATURE_NAMES))
    with c3:
        st.metric("Avg MPG", f"{df[TARGET].mean():.1f}")
    with c4:
        st.metric("MPG Range", f"{df[TARGET].min():.0f} - {df[TARGET].max():.0f}")

elif page == "📊 Small Data Showdown":
    st.markdown("<h1 style='margin-bottom:0;'>📊 Small Data Showdown</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Howso vs Random Forest — see how each performs when data is scarce.</p>", unsafe_allow_html=True)

    max_size = min(200, len(df))
    train_sizes = [10, 20, 30, 50, 75, 100, 150, max_size]
    train_sizes = [s for s in train_sizes if s <= max_size]

    with st.spinner("Running comparison across training sizes..."):
        results_df = run_small_data_comparison(df, tuple(train_sizes))

    if results_df.empty:
        st.warning("Not enough data for comparison.")
    else:
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("<div class='section-header'>⚡ Key Insight</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#f0f2ff;padding:1rem;border-radius:10px;border-left:4px solid #667eea;">
            Howso excels on <b>small datasets</b> because its instance-based learning doesn't need massive amounts of data to generalize. Traditional models like Random Forest need more data to avoid overfitting.
            </div>
            """, unsafe_allow_html=True)
            best_howso = results_df.dropna(subset=['howso_r2'])
            if not best_howso.empty:
                st.metric("Howso R² (smallest data)", f"{best_howso['howso_r2'].values[0]:.3f}")
            best_rf_small = results_df[results_df['train_size'] <= 30]['rf_r2'].mean()
            st.metric("Avg RF R² (≤30 samples)", f"{best_rf_small:.3f}")

        with col1:
            fig = go.Figure()
            mask_h = results_df['howso_r2'].notna()
            if mask_h.any():
                fig.add_trace(go.Scatter(
                    x=results_df.loc[mask_h, 'train_size'],
                    y=results_df.loc[mask_h, 'howso_r2'],
                    mode='lines+markers',
                    name='Howso',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, symbol='diamond'),
                ))
            fig.add_trace(go.Scatter(
                x=results_df['train_size'],
                y=results_df['rf_r2'],
                mode='lines+markers',
                name='Random Forest',
                line=dict(color='#ff6b6b', width=3, dash='dash'),
                marker=dict(size=10, symbol='circle'),
            ))
            fig.update_layout(
                title='R² Score vs Training Set Size',
                xaxis_title='Training Samples',
                yaxis_title='R² Score',
                yaxis_range=[-0.5, 1.0],
                hovermode='x unified',
                legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
                template='plotly_white',
                height=450,
            )
            fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>📋 Detailed Results</div>", unsafe_allow_html=True)
        display = results_df.copy()
        display.columns = ['Train Size', 'Howso R²', 'Howso MAE', 'RF R²', 'RF MAE', 'Test Samples']
        st.dataframe(display, hide_index=True, use_container_width=True)

elif page == "🔄 Synthetic Data Quality":
    st.markdown("<h1 style='margin-bottom:0;'>🔄 Synthetic Data Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Generate synthetic data with Howso and compare fidelity against the original.</p>", unsafe_allow_html=True)

    if not HOWSO_AVAILABLE:
        st.info("💡 Install Howso Engine to generate synthetic data: `pip install howso-engine`")
        st.markdown("<div class='section-header'>📊 Original Data Summary</div>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    else:
        with st.spinner("Generating synthetic data with Howso... (this may take ~30s)"):
            try:
                trainee = train_howso_targetless(df)
                if trainee is None:
                    st.error("Howso targetless training failed. Check logs.")
                    st.stop()

                synth_result = trainee.react(
                    action_features=df.columns.tolist(),
                    desired_conviction=10,
                    generate_new_cases='always',
                    num_cases_to_generate=len(df),
                )
                synth_df = synth_result['action']

                int_cols = df.select_dtypes(include=['int64', 'int32', 'int8']).columns
                for col in int_cols:
                    if col in synth_df.columns:
                        try:
                            synth_df[col] = synth_df[col].round().astype(int)
                        except Exception:
                            pass

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='section-header'>📊 Original Data</div>", unsafe_allow_html=True)
                    st.dataframe(df.describe().round(2), use_container_width=True)
                with col2:
                    st.markdown("<div class='section-header'>📊 Synthetic Data</div>", unsafe_allow_html=True)
                    st.dataframe(synth_df.describe().round(2), use_container_width=True)

                st.markdown("<div class='section-header'>📈 Distribution Comparison</div>", unsafe_allow_html=True)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols_per_row = 3
                rows_needed = (len(num_cols) + cols_per_row - 1) // cols_per_row
                fig = make_subplots(
                    rows=rows_needed, cols=cols_per_row,
                    subplot_titles=num_cols,
                    vertical_spacing=0.12, horizontal_spacing=0.08,
                )
                for i, col in enumerate(num_cols):
                    row = i // cols_per_row + 1
                    col_pos = i % cols_per_row + 1
                    fig.add_trace(
                        go.Histogram(x=df[col], name='Original', marker_color='#667eea', opacity=0.7, legendgroup='original', showlegend=(i == 0)),
                        row=row, col=col_pos,
                    )
                    if col in synth_df.columns:
                        fig.add_trace(
                            go.Histogram(x=synth_df[col], name='Synthetic', marker_color='#ff6b6b', opacity=0.6, legendgroup='synth', showlegend=(i == 0)),
                            row=row, col=col_pos,
                        )
                fig.update_layout(height=200 * rows_needed, barmode='overlay', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='section-header'>🔗 Correlation Comparison</div>", unsafe_allow_html=True)
                c1_h, c2_h = st.columns(2)
                with c1_h:
                    st.markdown("**Original Data**")
                    corr_orig = df[num_cols].corr()
                    fig_orig = px.imshow(corr_orig, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto', height=400)
                    fig_orig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_orig, use_container_width=True)
                with c2_h:
                    st.markdown("**Synthetic Data**")
                    synth_num_cols = [c for c in num_cols if c in synth_df.columns]
                    corr_synth = synth_df[synth_num_cols].corr()
                    fig_synth = px.imshow(corr_synth, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto', height=400)
                    fig_synth.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_synth, use_container_width=True)

                diff = (corr_orig - corr_synth.reindex_like(corr_orig)).abs()
                triu = np.triu_indices_from(diff.values, k=1)
                mean_diff = diff.values[triu].mean() if len(diff) > 1 and triu[0].size > 0 else 0

                st.markdown("<div class='section-header'>📊 Fidelity Metrics</div>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Mean Corr Diff", f"{mean_diff:.4f}")
                with m2:
                    orig_mean = df[num_cols].mean()
                    synth_avail = [c for c in num_cols if c in synth_df.columns]
                    synth_mean = synth_df[synth_avail].mean()
                    mean_bias = (orig_mean - synth_mean).abs().mean()
                    st.metric("Mean Feature Bias", f"{mean_bias:.3f}")
                with m3:
                    orig_std = df[num_cols].std()
                    synth_std = synth_df[synth_avail].std()
                    std_bias = (orig_std - synth_std).abs().mean()
                    st.metric("Mean Std Diff", f"{std_bias:.3f}")
                with m4:
                    st.metric("Synthetic Samples", len(synth_df))

            except Exception as e:
                st.error(f"Synthetic data generation failed: {e}")
                st.markdown(f"""
                <div style="background:#fff3cd;padding:1rem;border-radius:10px;">
                    <b>💡 Error details:</b> {e}
                </div>
                """, unsafe_allow_html=True)
