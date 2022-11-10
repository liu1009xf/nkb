
import requests
from bs4 import BeautifulSoup as bs
import bs4
import re
import datetime as dt
from tqdm import tqdm
from .util import TRACK_CODE_MAP,CODE_TRACK_MAP
import time

def get_race_id_list(start, end, sleep=0):
    res = list()
    dates = [start + dt.timedelta(n) for n in range((end-start).days)]
    for d in tqdm(dates, total=len(dates)):
        res+=get_race_id_list_from_date(d)
        time.sleep(sleep)
    return res

def get_race_id_list_from_date(today):
    date = f'{today.year:04}{today.month:02}{today.day:02}'
    url = 'https://db.netkeiba.com/race/list/' + date
    html = requests.get(url)
    html.encoding = "EUC-JP"
    soup = bs4.BeautifulSoup(html.text, "html.parser")
    race_list = soup.find('div', attrs={"class": 'race_list fc'})
    if race_list is None:
        return list()
    a_tag_list = race_list.find_all('a')  # type: ignore
    href_list = [a_tag.get('href') for a_tag in a_tag_list]
    race_id_list = list()
    for href in href_list:
        for race_id in re.findall('[0-9]{12}', href):
            race_id_list.append(race_id)
    return list(set(race_id_list))

def get_race_meta_by_date(today):
    url = f'https://www.jra.go.jp/keiba/calendar{today.year:04}/{today.year:04}/{today.month}/{today.month:02}{today.day:02}.html'
    html = requests.get(url)
    html.encoding = "Shift_JIS"
    soup = bs4.BeautifulSoup(html.text, "html.parser")
    all_races = list()
    race_list = soup.find('div', attrs={"class": 'grid'}).find_all('div', attrs={"class": 'cell'})
    race_rounds = {x.find('div', attrs={"class": 'main'}).contents[0]:x.find('tbody').find_all('tr') 
        for x in race_list if not x.find('div', attrs={"class": 'main'}) is None}
    for k,v in race_rounds.items():
        r=re.search(r"回(.*?)日",k).group(1)
        r=re.search(r'[^\d.]+',r).group(0)
        c,d = re.findall(r'\d+', k)
        all_races = all_races + [{'track_code':CODE_TRACK_MAP[r],'trace_name':r, 'time':c,'day':d, 
          'round':x.find('th').contents[0], 'name':x.find('p', attrs={'class':'race_class'}).contents[0], 
          'distance':x.find('p', attrs={'class':'race_cond'}).find('span', attrs={'class':'dist'}).contents[0],
          'datetime':dt.datetime.combine(today, dt.datetime.strptime(x.find('td', attrs={'class':'time'}).contents[0], "%H時%M分").time())} for x in v]
    all_races = [x|{'_id':f'{today.year}{x.get("track_code")}{int(x.get("time")):02}{int(x.get("day")):02}{int(x.get("round")):02}'}for x in all_races]
    return all_races

def get_future_race_id_by_date(today):
    url = f'https://www.jra.go.jp/keiba/calendar{today.year:04}/{today.year:04}/{today.month}/{today.month:02}{today.day:02}.html'
    html = requests.get(url)
    html.encoding = "Shift_JIS"
    soup = bs4.BeautifulSoup(html.text, "html.parser")
    all_races = list()
    race_list = soup.find('div', attrs={"class": 'grid'}).find_all('div', attrs={"class": 'cell'})
    race_rounds = {x.find('div', attrs={"class": 'main'}).contents[0]:x.find('tbody').find_all('tr') 
        for x in race_list if not x.find('div', attrs={"class": 'main'}) is None}
    for k,v in race_rounds.items():
        r=re.search(r"回(.*?)日",k).group(1)
        r=re.search(r'[^\d.]+',r).group(0)
        c,d = re.findall(r'\d+', k)
        all_races = all_races + [{'track_code':CODE_TRACK_MAP[r],'trace_name':r, 'time':c,'day':d, 'round':x.find('th').contents[0]} for x in v]
    all_races = [f'{today.year}{x.get("track_code")}{int(x.get("time")):02}{int(x.get("day")):02}{int(x.get("round")):02}' for x in all_races]
    return all_races