import datetime as dt
import bs4
import requests
import re
TRACK_CODE_MAP = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
CODE_TRACK_MAP = {v:k for k,v in TRACK_CODE_MAP.items()}

def get_race_date(race_id:str) -> dt.datetime:
    url= f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu'
    return dt.datetime.strptime(_get_soup(url).find('dd', attrs={'class':'Active'}).find('a').contents[0], '%m月%d日')
def get_race_start_time(race_id:str) -> dt.time:
    url= f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu'
    return dt.datetime.strptime(
        ':'.join(re.findall(r'\d+', 
        _get_soup(url).find('div', attrs={'class':'RaceData01'}).contents[0])), '%H:%M').time()


def _get_soup(url:str) -> bs4.BeautifulSoup:
    html = requests.get(url)
    html.encoding = "EUC-JP"
    return bs4.BeautifulSoup(html.text, "html.parser")