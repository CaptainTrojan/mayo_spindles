import pandas as pd
import os
import plotly.express as px
dir = 'activelearning_annotations'

for file in os.listdir(dir):
    df = pd.read_csv(os.path.join(dir, file))
    
    fig = px.histogram(df, x='type', title=file)
    fig.show()