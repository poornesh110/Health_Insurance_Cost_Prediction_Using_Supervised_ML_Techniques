#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


import pickle
model = pickle.load(open('model_pkl', 'rb'))


# In[6]:


def predict_insurance(age,sex,bmi,children,smoker,region):
    cols = ['age','bmi','children','region_northeast','region_northwest',
            'region_southeast','region_southwest','sex_female','sex_male','smoker_no','smoker_yes']
    df = pd.DataFrame(columns = cols)
    df.loc[0,'age'] = age
    df.loc[0,'bmi'] = bmi
    df.loc[0,'children'] = children
    df.loc[0,'region_northeast'] = 1 if region == 'northeast' else 0
    df.loc[0,'region_northwest'] = 1 if region == 'northwest' else 0
    df.loc[0,'region_southeast'] = 1 if region == 'southeast' else 0
    df.loc[0,'region_southwest'] = 1 if region == 'southwest' else 0
    df.loc[0,'sex_female'] = 1 if sex == 'female' else 0
    df.loc[0,'sex_male'] = 1 if sex == 'male' else 0
    df.loc[0,'smoker_no'] = 1 if smoker == 'no' else 0
    df.loc[0,'smoker_yes'] = 1 if smoker == 'yes' else 0
    
    predicted_cost = model.predict(df)
    return ("{:.4f}".format(predicted_cost[0]))


# In[7]:


import gradio as gr
from gradio.components import Textbox, Checkbox
from gradio.components import Dropdown


age_input = "number"
bmi_input = "number"
region_input = Dropdown(choices=["northeast", "northwest", "southeast","southwest"], label="Select your region")
smoker_input = Dropdown(choices=["yes", "no"], label="Do you Smoke?")
sex_input = Dropdown(choices=["male", "female"], label="Select your Gender")
children_input = "number"

iface =gr.Interface(fn = predict_insurance,
                    inputs = [age_input,sex_input,bmi_input,children_input,smoker_input,region_input],
                    outputs = [Textbox(label="Prediction")],
                    title = "Health Insurance Cost Prediction based on multiple factors",
                    description = "Please Provide the below details")
iface.launch()


# In[ ]:





# In[ ]:




