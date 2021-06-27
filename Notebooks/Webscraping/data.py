# coding=utf-8
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
import os

# Opciones de navegación
options =  webdriver.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('--disable-extensions')
options.add_argument("no-sandbox")
options.add_argument("--headless")
driver_path = '/usr/bin/chromedriver'

driver = webdriver.Chrome(driver_path, chrome_options=options)

# Iniciarla en la pantalla 2
driver.set_window_position(2000, 0)
driver.maximize_window()

# Inicializamos el navegador

#Bloomberg
web_url_1 = 'https://www.bloomberg.com/news/articles/2021-02-28/with-goldman-hires-mcmillon-moves-closer-to-bank-of-walmart?srnd=premium' 
#Yahoo Finance
web_url_2 = 'https://finance.yahoo.com/quote/%5EGSPC/history?period1=1410912000&period2=1620345600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
driver.get(web_url_2)
time.sleep(1)
elem = driver.find_element_by_tag_name("body")
no_of_pagedowns = 130

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    no_of_pagedowns-=1

if(no_of_pagedowns == 0):
   r = requests.get(web_url_2)
   soup = BeautifulSoup(r.text,'html.parser')
   page_title = soup.title.text

   print(page_title)
   rows = []
   cols = []
   df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

   table = driver.find_element_by_tag_name('table')
   table_rows = table.find_elements(By.TAG_NAME, "tr") # get all of the rows in the table
   print("Total rows: ", len(table_rows))
   row_counter = 0
   for row in table_rows:
      # Get the columns (all the column 2)        
      cells = row.find_elements(By.TAG_NAME, "td") #note: index start from 0, 1 is col 2
      for t in cells:
         if(hasattr(t, 'text')):
            if(len(t.text) <= 15):
               cols.append(t.text)
      if(len(cols)>1):
         rows.append(cols)
         tempDF = pd.DataFrame([cols],columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
         df = pd.concat([df,tempDF])
      cols = []
      row_counter += 1
      print("Rows left: ", len(table_rows)-row_counter)

   print ("tail: \n",df.tail(5))
   df.to_csv(r'/home/ubuntu/Notebooks/Webscraping/Data/SP_500.csv', index=False)
   driver.quit()
