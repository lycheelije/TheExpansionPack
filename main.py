import os
import pandas as pd
from huggingface_hub import hf_hub_download

def valid_row(row):
  # word blacklist file is located at ...
  # blacklist = open('...', 'r').read()
  # if row.column NOT in blacklist:                        # TODO
  #   return true
  # else:
    return false

def filter_dataset(file_path):                 # accepts the file path of the un-filtered dataset
  in_file = open(file_path, 'r')
  file_name = os.path.basename(file_path)
  out_file = open('filtered_'+file_name, 'w')
  data = in_file.read()
  for row in data:
    if valid_row(row):                      # sends the row to be validated
      out_file.write(row)
  return('filtered_'+file_path)                # returns the path of the now filtered data set

def new_func(parameter):
  #process
  return(false)

#from flask import Flask
#
#app = Flask(__name__)
#
#@app.route('/')
#def hello_world():
#    return 'Hello, World!'
#
#if __name__ == '__main__':
#    app.run()
