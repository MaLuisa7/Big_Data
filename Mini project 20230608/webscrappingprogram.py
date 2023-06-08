# Import the required libraries.
#https://medium.com/geekculture/web-scraping-game-reviews-with-python-and-beautifulsoup-b1372baa479c
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv

 
# extraer el archivo HTML y se crea un objeto de BeautifulSoup .
url = 'https://www.metacritic.com/game/switch/super-mario-3d-world-+-bowsers-fury/user-reviews'

user_agent = {'User-agent': 'Mozilla/5.0'}

def get_page_contents(url):
    page = requests.get(url, headers = user_agent)
    return BeautifulSoup(page.text, 'html.parser')

soup = get_page_contents(url)

#creamos las listas de las columnas que nos interesan
names = []
for links in soup.find_all('div', class_='name'):
     name = links.get_text().strip()
     names.append(name)
dates = []
for links in soup.find_all('div', class_='date'):
     date = links.get_text()
     dates.append(date)
ratings = []   
for links in soup.find_all('div', class_='metascore_w user medium game positive indiv'):
     score = links.get_text()
     ratings.append(score)
reviews = []   
for links in soup.find_all('span', class_='blurb blurb_expanded'):
     review = links.get_text()
     reviews.append(review)

for links in soup.find_all('span', class_='blurb blurb_collapsed'):
     review = links.get_text()
     reviews.append(review)
     
# creamos el diccionario y despues el df que exportaremso con los datos
games_dict = {'Name': names, 'Date': dates, 'Rating': ratings, 'Review': reviews}
print(len(names), len(dates), len(ratings), len(reviews))
game = pd.DataFrame.from_dict(games_dict, orient='index')
games = game.transpose()

games.head(4)
games.to_csv('reviews.csv', index=False, header=True)
reviews = pd.read_csv('reviews.csv', lineterminator='\n')