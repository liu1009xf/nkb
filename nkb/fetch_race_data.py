from collections import ChainMap
import itertools
from typing import Any, List, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
import requests
from bs4 import BeautifulSoup as bs
import bs4
import re
import datetime as dt

import abc

from tqdm.notebook import tqdm

def _racetimestr2timedelta(time_str:str) -> dt.time:
    times = time_str.split('.')
    times1 = times[0].split(':')
    time = dt.timedelta(seconds=int(times1[0])*60+int(times1[1]),
                        microseconds=int(times[1])*100000).total_seconds()
    return time

class DataLoader:
    def __init__(self, 
                id:str, 
                base_url:str='https://db.netkeiba.com/',
                header = 0
                ) -> None:
        self.id = id
        self.url = f'{base_url}{self.PATH_SUFFIX()}{id}'
        html = requests.get(self.url)
        html.encoding = 'EUC-JP'
        self.soup = bs4.BeautifulSoup(html.text, 'html.parser')
        tb_args = dict()
        if header:
            tb_args = tb_args|{'header':header}
        self.tables=pd.read_html(html.text, **tb_args)
        self.data = None
        try:
          self.data = self.fetch_data()
        except Exception as e:
          self.data = self._back_up_fetch_data(e)
        
        self.data = self.data.astype(self.COL_TYPES())
        self.process_data()

    @classmethod
    @abc.abstractmethod
    def DATA_TYPE(cls) -> str:
        raise NotImplementedError("Please Implement Data Type")

    @classmethod
    def PATH_SUFFIX(cls)->str:
        return f'{cls.DATA_TYPE()}/'
    
    @classmethod
    def ID_PREFIX(cls)->str:
        return cls.DATA_TYPE()

    @classmethod
    @abc.abstractmethod
    def COL_TYPES(cls) -> str:
        raise NotImplementedError("Please Implement Data Type")
    
    @abc.abstractmethod
    def process_data(self ) -> None:
        pass

    @abc.abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        raise NotImplementedError("Please Implement fetch_data")
    
    def unique_ids(self) -> List[int]:
        return range(len(self.data))

    def save(self,db, ignore_conflict=True)-> None:
        data = self.data
        data['_id'] = self.unique_ids()
        data = data.to_dict('records')
        col = db[self.DATA_TYPE()]
        if(ignore_conflict):
            try:
                col.insert_many(data, ordered=False)
            except:
                pass
        else:
            col.insert_many(data)
    
    def _back_up_fetch_data(self, err):
        # traceback.print_exc(err)
        pass
            

class PayoffDataLoader(DataLoader):
    def __init__(self, 
                id:str, 
                base_url:str='https://db.sp.netkeiba.com/') -> None:
        super().__init__(id, base_url, header = None)

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'payoff'

    @classmethod
    def PATH_SUFFIX(cls)->str:
        return 'race/'
    
    def unique_ids(self) -> List[int]:
        return self.data[['race_id', 'type', 'strike']].apply(lambda x:f'{x["race_id"]}_{x["type"]}_{x["strike"]}', axis=1)
    
    @classmethod
    def ID_PREFIX(cls)->str:
        return cls.PATH_SUFFIX()
    
    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'type'   :str,
                'strike' :str,
                'payoff' :'int64',
        }
    
    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_data(self) -> pd.DataFrame:
        df=self.tables[1]
        df.columns=['type','strike','payoff','popularity']
        df=df.drop('popularity', axis=1)
        df['payoff'] = pd.to_numeric(df['payoff'].apply(lambda x: "".join(re.findall(r'\d+', x))) , errors='coerce')       
        return self.append_race_id(df)
    
    def _back_up_fetch_data(self, err)->pd.DataFrame:
        return pd.DataFrame([{'type':x.find('th').contents[0] if x.find('th') else np.nan,
                    'strike':x.find('td', attrs={'class':'Result'}).contents[0],
                    'payoff':int(x.find('td', attrs={'class':'Payout'}).contents[0].replace('円', '').replace(',', '')),
                    'race_id':'202005040601' } for x in self.soup.find_all('table')[1].find_all('tr')]).fillna(method='ffill')


class ShutsubaDataLoader(DataLoader):

    def __init__(self, 
                id:str, 
                base_url:str='https://race.netkeiba.com/') -> None:
        super().__init__(id, base_url)

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'race'

    @classmethod
    def PATH_SUFFIX(cls) -> str:
        return 'race/shutuba.html?race_id='
    
    def unique_ids(self) -> List[int]:
        return self.id

    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'lane_num'          :'int64',
                'horse_num'         :'int64',
                'load_weight'       :'float64',
                'sex'               :str,
                'age'               :'int64',
                'horse_weight'      :'int64',
                'horse_weight_diff' :'int64',
                'norm_horse_weight' :'float64',
                'norm_load_weight'  :'float64',
                'race_id'           :str,
                'distance'          :str,
                'is_hindrance'      :'int64',
                'field_type'        :str,
                'direction'         :str,
                'ground_condition'  :str, 
                'weather'           :str, 
                'date'              :str, 
                'start_time'        :str, 
                'horse_id'          :str, 
                'jockey_id'         :str, 
                'trainer_id'        :str}
    
    def fetch_data(self) -> pd.DataFrame:
        df = self.fetch_result()
        return df 

    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_result(self):
        df = self.tables[0]
        df.columns = [x.replace(' ', '') for x in df.columns]
        df = df.drop(df.index[0])
        df = self.append_race_id(df)
        df = pd.merge(df, self.fetch_race_meta(), on='race_id')
        df['horse_id'] = self.fetch_horse_id()
        df['jockey_id'] = self.fetch_jockey_id()
        df['trainer_id'] = self.fetch_trainer_id()
        
        df["sex"] = df["性齢"].map(lambda x: str(x)[0])
        df["age"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        df["horse_weight"] = df["馬体重(増減)"].str.split("(", expand=True)[0]
        df=df[~(df["horse_weight"].astype(str).str.contains('\\D'))]
        df["horse_weight"] = df["horse_weight"].astype(int)
        df["horse_weight_diff"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1].astype(int)

        df.drop(["印","馬名", "性齢","騎手","厩舎","馬体重(増減)", "Unnamed: 9", "お気に入り馬", "人気","お気に入り馬.1"], axis=1, inplace=True)
        
        col_names=['lane_num','horse_num', 'load_weight']
        col_names+=['race_id','is_hindrance', 'direction','date','start_time']
        col_names+=[ 'field_type','distance','weather','ground_condition', 'horse_id']
        col_names+=['jockey_id', 'trainer_id']
        col_names+=['sex','age','horse_weight', 'horse_weight_diff']
        df.columns=col_names
        df["load_weight"] = df["load_weight"].astype(float)
        normalize_cols = ['horse_weight', 'load_weight']
        df[[f'norm_{x}' for x in normalize_cols]] = df[normalize_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return df

    def fetch_race_meta(self) -> pd.DataFrame:
        tag = self.soup.find('div', attrs={'class': 'RaceData01'})
        date_tag = self.soup.find("dd", attrs={"class": "Active"}).find("a")
        if isinstance(tag, bs4.element.Tag) and isinstance(date_tag, bs4.element.Tag):
            res=dict()
            def get_field_type(input, options):
                fields = []
                for ch in input:
                    if ch in options:
                        fields.append(ch)
                return '/'.join(fields)
            dir_options = ["左", "右", '内', '外']
            field_options = ["芝", "ダ"]
            vals = [x if isinstance(x,bs4.element.NavigableString) else x.contents for x in tag if not isinstance(x,bs4.element.Comment)]
            vals = [x if isinstance(x, str) else x[0] for x in vals if x and x!='\n']
            vals = [x for x in list(itertools.chain.from_iterable([x.split("\n") for x in vals])) if x]
            res['is_hindrance'] = int(False)
            res['direction'] = ''
            res['date']=f'{self.id[:4]}年{date_tag.contents[0]}'
            for val in vals:
                if isinstance(val, str):
                    if '発走' in val:
                        res['start_time'] = ':'.join(re.findall(r'\d+', val))
                    if '馬場' in val:
                        res['ground_condition'] = val.split(":")[1]
                    if '天候' in val:
                        res['weather'] = val.split(":")[1]
                    if any([x in val for x in dir_options]):
                        res['direction']= get_field_type(val,dir_options)
                    if any([x in val for x in field_options]):
                        res['field_type'] = get_field_type(val, field_options)
                        if len(re.findall(r'\d+', val)) == 1:
                            res['distance'] = re.findall(r'\d+', val)
                    if "障" in val:
                        res['distance'] = re.findall(r'\d+', val)[0]
                        res['is_hindrance'] = int(True)
            df= pd.DataFrame(res, index=[0])
            return self.append_race_id(df)
        else:
            raise RuntimeError("can not find the right tag")

    def _fetch_id_from_summary(self, field:str='span', cls:str="HorseName") -> pd.DataFrame:
        return [re.findall(r'\d+', y['href'])[0] for y in [x.find('a') for x in self.soup.find_all(field, attrs={'class':cls})] if y]

    def fetch_jockey_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='td', cls='Jockey')
    
    def fetch_horse_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary()
    
    def fetch_trainer_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='td',cls='Trainer')

    def process_data(self) -> None:
        self.data['date']= self.data['date'].apply(lambda x: dt.datetime.strptime(x, '%Y年%m月%d日').date())
        self.data['start_time']= self.data['start_time'].apply(lambda x:dt.datetime.strptime(x, '%H:%M').time())
        self.data['start_time']= self.data[['date', 'start_time']].apply(lambda x: dt.datetime.combine(x['date'], x['start_time']),axis=1)
        self.data['date']= self.data['date'].apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))

    def save(self,db, ignore_conflict=True)-> None:
        col = db['rt_shutsuba']
        data = {'_id':self.unique_ids(), 'data':self.data.to_dict('records')}
        col.insert_one(data)

class RaceDataLoader(DataLoader):

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'race'
    
    def unique_ids(self) -> List[int]:
        return self.data[['race_id', 'horse_id']].apply(lambda x:f'{x["race_id"]}_{x["horse_id"]}', axis=1)

    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'rank'              :'int64',
                'lane_num'          :'int64',
                'horse_num'         :'int64',
                'horse_name'        :str,
                'load_weight'       :'float64',
                'jockey_name'       :str,
                'time'              :str,
                'rank_first_odd'    :'float64',
                'popularity'        :'int64',
                'sex'               :str,
                'age'               :'int64',
                'horse_weight'      :'int64',
                'horse_weight_diff' :'int64',
                'norm_horse_weight' :'float64',
                'norm_load_weight'  :'float64',
                'race_id'           :str,
                'distance'          :str,
                'is_hindrance'      :'int64',
                'field_type'        :str,
                'direction'         :str,
                'ground_condition'  :str, 
                'weather'           :str, 
                'date'              :str, 
                'start_time'        :str, 
                'horse_id'          :str, 
                'jockey_id'         :str,
                'owner_name'        :str, 
                'owner_id'          :str,
                'trainer_name'      :str, 
                'trainer_id'        :str,
                'prize'             :"float64"}
    
    def fetch_data(self) -> pd.DataFrame:
        df = self.fetch_result()
        return df 

    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_result(self):
        df = self.tables[0]
        df.columns = [x.replace(' ', '') for x in df.columns]
        df = self.append_race_id(df)
        df = df.rename(columns={'馬名':'horse_name','騎手':'jockey_name'})
        df = pd.merge(df, self.fetch_race_meta(), on='race_id')
        # df = pd.merge(df, self.fetch_horse_id(), on='horse_name')
        df = df.join(self.fetch_horse_id()['horse_id'])
        df = df.join(self.fetch_jockey_id()['jockey_id'])
        df = df.join(self.fetch_owner_id())
        df = df.join(self.fetch_trainer_id())
        df['prize'] = self.fetch_prizes()
        df = df[~(df["着順"].astype(str).str.contains('\\D'))]
        df = deepcopy(df)
        df["着順"] = df["着順"].astype(int)
        
        df["sex"] = df["性齢"].map(lambda x: str(x)[0])
        df["age"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        df["horse_weight"] = df["馬体重"].str.split("(", expand=True)[0]
        df=df[~(df["horse_weight"].astype(str).str.contains('\\D'))]
        df["horse_weight"] = df["horse_weight"].astype(int)
        df["horse_weight_diff"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        df["単勝"] = df["単勝"].astype(float)

        df.drop(["着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)
        
        #着順,枠番,馬番,馬名,斤量,騎手,単勝,人気	
        col_names=['rank','lane_num','horse_num', 'horse_name']
        col_names+=['load_weight', 'jockey_name', 'time', 'rank_first_odd', 'popularity']
        col_names+=['race_id','distance','is_hindrance', 'field_type', 'direction', 'ground_condition']
        col_names+=['weather','date','start_time', 'horse_id']
        col_names+=['jockey_id','owner_name','owner_id', 'trainer_name', 'trainer_id', 'prize']
        col_names+=['sex','age','horse_weight', 'horse_weight_diff']
        df.columns=col_names
        normalize_cols = ['horse_weight', 'load_weight']
        df[[f'norm_{x}' for x in normalize_cols]] = df[normalize_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return df

    def fetch_returns(self) -> pd.DataFrame:
        df=self.tables[1]
        df.columns=['type','result_num','payoff','popularity']
        df = df.groupby(['type']).agg(tuple).applymap(list).reset_index()
        df['result']= df[[x for x in df.columns if x!='type']].to_dict(orient='records')

        df=df.drop([x for x in df.columns if not x in ['type','result']], axis=1)
        df.index = pd.Index([0]*len(df))
        df = df.pivot(columns=['type'])
        df.columns = df.columns.droplevel()        
        return self.append_race_id(df)

    def fetch_race_meta(self) -> pd.DataFrame:
        tag = self.soup.find('div', attrs={'class': 'data_intro'})
        res = dict()
        def decompose(val):
            val = val.replace('m', '')
            res= dict()
            res['distance'] = re.findall(r'\d+', val)[0]
            field=val.replace(res['distance'], "")
            values = re.findall(r'\w', field)
            def get_field_type(input, options):
                fields = []
                for ch in input:
                    if ch in options:
                        fields.append(ch)
                return '/'.join(fields)
            if values[0] =="障":
                res['is_hindrance'] = 1
                field_type = values[1:]
                res['field_type'] = get_field_type(field_type, ["芝", "ダ"])
            else:
                res['is_hindrance'] = 0
                res['field_type'] = get_field_type(values, ["芝", "ダ"])
            res['direction'] = get_field_type(values, ["左", "右", '内', '外'])
            return res
        if isinstance(tag, bs4.element.Tag):
            infos=re.findall(r'\w+',
                            tag.find_all('p')[0].text + tag.find_all('p')[1].text)
            infos += re.findall(r'\w+:\w+', tag.find_all('p')[0].text)
            idx = [i for i,x in enumerate(infos) if re.search( r'\d+m', x)][0]
            res = decompose(''.join(infos[:idx+1]))
            ground_condition = []
            for info in infos:
                if info in ['良','稍重', '重', '不良']:
                    ground_condition.append(info)
                if info in ['曇', '晴', '雨', '小雪', '雪','小雨']:
                    res['weather']= info
                if '年' in info and '月' in info:
                    res['date']= info
                if ':' in info:
                    res['start_time']= info
                res['ground_condition'] = '/'.join(ground_condition)
            df = pd.DataFrame(res,index=[0])
            return  self.append_race_id(df)
        else:
            raise RuntimeError("can not find the right tag")

    def _fetch_id_from_summary(self, field='horse') -> pd.DataFrame:
        tag = self.soup.find('table', attrs={'summary': 'レース結果'})
        infos=None
        names = list()
        fields = list()
        df = None
        if isinstance(tag, bs4.element.Tag):
            infos=tag.find_all('a', attrs={'href':re.compile(f'^/{field}')})
            for info in infos:
                names.append(info['title'])
                fields.append(re.findall(r'\d+', info['href'])[0])
            df=pd.DataFrame(list(zip(names, fields)),
                            columns =[f'{field}_name', f'{field}_id'])
        else:
            raise RuntimeError("Tag not fund")
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Data Frame is None")
        return df

    def fetch_jockey_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='jockey')
    
    def fetch_horse_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='horse')
    
    def fetch_owner_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='owner')
    
    def fetch_trainer_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='trainer')
    
    def fetch_prizes(self)->list:
        tag = self.soup.find('table', attrs={'summary': 'レース結果'})
        res = list()
        if isinstance(tag, bs4.element.Tag):
            res = [x.find_all('td')[-1].contents for x in tag.find_all('tr') if len(x.find_all('td'))>0]
            res = [0.0 if not x else float(x[0].replace(",","")) for x in res]
        else:
            raise RuntimeError("Tag not fund")
        return res
    
    def get_all_horse_id(self)->list[str]:
        return []

    def process_data(self) -> None:
        self.data['time'] = self.data['time'].apply(lambda x: _racetimestr2timedelta(x))
        self.data['date']= self.data['date'].apply(lambda x: dt.datetime.strptime(x, '%Y年%m月%d日').date())
        self.data['start_time']= self.data['start_time'].apply(lambda x:dt.datetime.strptime(x, '%H:%M').time())
        self.data['start_time']= self.data[['date', 'start_time']].apply(lambda x: dt.datetime.combine(x['date'], x['start_time']),axis=1)
        self.data['date']= self.data['date'].apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))


class HorsePedDataLoader(DataLoader):
    def __init__(self, 
                id:str, 
                base_url:str='https://db.netkeiba.com/') -> None:
        super().__init__(id, base_url, header = None)

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'horse_ped'

    @classmethod
    def PATH_SUFFIX(cls)->str:
        return '/horse/ped/'
    
    def unique_ids(self) -> List[int]:
        return self.data['horse_id']
    
    @classmethod
    def ID_PREFIX(cls)->str:
        return 'horse'
    
    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'horse_id'   :str,
                'father'     :str,
                'mother'     :str,
                'fgfather'   :str,
                'fgmother'   :str,
                'mgfather'   :str,
                'mgmother'   :str,
                'father_ped' :str,
                'mother_ped' :str
        }
    
    def append_horse_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['horse_id'] = self.id
        return df

    def fetch_data(self) -> pd.DataFrame:
        df=self.tables[0]
        def get_horse_name(l, index):
            l=l.loc[l.shift(-1) != l]
            l1 = [x.split(' ') for x in l][index]
            idx = [i for i, item in enumerate(l1) if re.search('\d{4}$', item)][0]
            return ' '.join(l1[:idx])
        name_dict = {
            'father':get_horse_name(df[0], 0),
            'mother':get_horse_name(df[0], 1),
            'fgfather':get_horse_name(df[1], 0),
            'fgmother':get_horse_name(df[1], 1),
            'mgfather':get_horse_name(df[1], 2),
            'mgmother':get_horse_name(df[1], 3),
            'fgffather':get_horse_name(df[2], 0),
            'fgfmother':get_horse_name(df[2], 1),
            'fgmfather':get_horse_name(df[2], 2),
            'fgmmother':get_horse_name(df[2], 3),
            'mgffather':get_horse_name(df[2], 4),
            'mgfmother':get_horse_name(df[2], 5),
            'mgmfather':get_horse_name(df[2], 6),
            'mgmmother':get_horse_name(df[2], 7)
        }
        tags = [x.find(lambda tag:tag.name=="a") for x in 
            list(itertools.chain(*[x.find_all('td') for x in self.soup.find('table', attrs={'summary':"5代血統表"}).find_all('tr')]))]
        tag_dict = dict(ChainMap(*[{' '.join([r for r in ' '.join([w for w in x.text.split('\n') if w]).split(' ') if r]):x['href']} for x in tags if x]))
        id_dict = {k:[x for x in tag_dict[v].split('/') if x][-1]for k,v in name_dict.items()}
        all_dict = {**id_dict, **dict(zip(['father_ped', 'mother_ped'], [x.split(' ')[-1] for x in df[0].unique()]))}
        df1 = pd.DataFrame(all_dict,index=[0])
        return self.append_horse_id(df1)


