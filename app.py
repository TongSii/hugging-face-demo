import time
import gradio as gr
from util import *
from fGAIN_loss import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#Setup inputs and outputs
inputs = [
    #inputs for getting a file
    gr.File(label="CSV file"),
    gr.Textbox(label="Google Colab file path"),
    #input for the structure of the data
    gr.CheckboxGroup(label="Does the data set have these?", choices=["Row Header", "Column Header"]),
    #input for the loss functions
    gr.CheckboxGroup(label="Loss functions", value="All", choices=["All", "Original", "Forward KL", "Reverse KL", "Pearsom Chi-Squared", "Squared Hellingr", "Jensen-Shannon"]),
    #input for the hyperparameters of the model
    gr.Slider(label="Missing Rate", maximum=1, value=.2, step=.01),
    gr.Number(label="Batch Size", value=128),
    gr.Slider(label="Hint Rate", maximum=1, value=.9, step=.01),
    gr.Number(label="Alpha", value=100),
    gr.Number(label="Iterations", value=10000)
]

#outputs for the best loss and the table with the results
outputs = [
    gr.Textbox(label="Best Loss"),
    gr.Dataframe(label="Loss Table")
]


#Function
def run_losses(csv_file, colabFilePath, header_list, loss_fn_list, miss_rate, batch_size, hint_rate, alpha, iterations):
  bestLossStr = "_"
  pd_loss_res = pd.DataFrame(columns=["Loss Function", "RMSE"])
  #making header inputs
  row_header_input = None if "Row Header" not in header_list else 0
  col_header_input = 'infer' if "Column Header" not in header_list else 0

  #load the data
  if csv_file != None:
    data = pd.read_csv(csv_file.name, delimiter=',', index_col=row_header_input, header=col_header_input) 
  elif colabFilePath != "":
    data = pd.read_csv(colabFilePath, delimiter=',', index_col=row_header_input, header=col_header_input) 
  else:
    return "Didn't input a file path or file", pd_loss_res 

  bestLossInt = 1e6
  #Run the data on all versions of the GAIN's loss
  if "All" in loss_fn_list:
    loss_fn_list = ["Original", "Forward KL", "Reverse KL", "Pearsom Chi-Squared", "Squared Hellingr", "Jensen-Shannon"]

  for loss_fn in loss_fn_list:
    _, rmse = main(data, miss_rate, int(batch_size), hint_rate, alpha, int(iterations), loss_fn)
    pd_loss_res = pd_loss_res.append({"Loss Function":loss_fn, "RMSE": rmse}, ignore_index=True)
    if rmse < bestLossInt:
      bestLossInt = rmse
      bestLossStr = f"Best Loss function is {loss_fn} with RMSE: {rmse}" 

  print(bestLossStr)
  print(pd_loss_res)
  return bestLossStr, pd_loss_res


#start the interface
gr.Interface(fn=run_losses, inputs=inputs, outputs=outputs).launch(debug=True)