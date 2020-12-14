import pandas as pd
import requests
from io import StringIO

def read_csv(url):
  url = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
  csv_raw = requests.get(url).text
  csv = StringIO(csv_raw)
  df = pd.read_csv(csv)
  return df

def reading_data():
    ss = read_csv("https://drive.google.com/file/d/1B5OdhqtRPjkhwouLG9DIJNQH5bXPmro1/view?usp=sharing")
    test = read_csv("https://drive.google.com/file/d/1hEa9980OEnMi8GR3bDik0YohBwuqRds2/view?usp=sharing")
    train = read_csv("https://drive.google.com/file/d/1AFucfssgUvmjNilr9V7VkB1AYpULJefx/view?usp=sharing")
    des = read_csv("https://drive.google.com/file/d/1wC4vw3WAZvPYLAtPBXRVkf2sE0guWGGc/view?usp=sharing")
    return ss, test, train, des
