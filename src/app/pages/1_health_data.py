import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from src.config import DATA_DIR, USER_ID_COLUMN, VARS, DATE_COLUMN,USER_ID
from src.utils.functions import find_episodes

user_data = pd.read_csv(DATA_DIR / "processed/clean_data_truncated.csv")
user_data = user_data[user_data[USER_ID_COLUMN] == USER_ID].drop(columns=[i for i in user_data.columns if i not in VARS + [DATE_COLUMN]])
user_data["date"] = pd.to_datetime(user_data["date"],yearfirst=True,dayfirst=False,format="%Y-%m-%d")

def make_date_fig(df,col_name:str,func:callable,y_label:str|None=None,episodes=True):
    if y_label is None:
        y_label=col_name
    fig = func(df,x="date",y=col_name,
               labels={"date":"Date",col_name:y_label})
    last_date = df.date.max()
    fig.update_xaxes(range=[last_date -pd.Timedelta(days=7),last_date+pd.Timedelta(days=1.5)])
    fig.update_yaxes(range=[max(df[col_name].min()-2,0),df[col_name].max() +1],fixedrange=True)

    if episodes:
        for episode in find_episodes(user_data):
            fig.add_vrect(
                x0=episode[0]-pd.Timedelta(days=0.5),
                x1=episode[1]+pd.Timedelta(days=0.5),
                fillcolor="red", 
                opacity=0.3, 
                layer="below", 
                line_width=0
            )
    return fig

    
def make_2col_date_plot(df,col1:str,col2:str, funcs:list[callable] = [px.bar,px.bar],labels:list[str|None] = [None,None],episodes=[True,True]):
    c1,c2 = st.columns(2)
    c1.plotly_chart(make_date_fig(df,col1,funcs[0],labels[0],episodes=episodes[0]))
    c2.plotly_chart(make_date_fig(df,col2,funcs[1],labels[1],episodes=episodes[1]))

def make_col_entry(entries:list[dict],level=2):
    for column,entry in zip(st.columns(len(entries)),entries):
        if "title" in entry:
            column.write(f"{'#'*level} {entry['title']}")
        if "content" in entry:
            c = entry["content"].replace("\n","\n\n")
            column.write(c)

#PAGE
st.write("# Your health data")
st.write("Here you can add your health data to see your patterns of migraines. Currently this functionality is not available, so instead you can see the patterns of a demo user.")
st.write(user_data)

ent1 = {"title":"Migraine severity","content":"Here you can see you can see your migraine severity over time. The app uses this data to gain insights into your patterns."}
ent2 = {"title":"Sleep","content":"Your sleep can affect the occurrence of migarine episodes - make sure you sleep long enough every night"}
ent3 = {"title":"Hydration","content":"Hydration can affect the severity of a migraine"}
ent4 = {"title":"Screen Time","content":"Increased screen time was linked to increased severity of a migraine"}
ent5 = {"title":"Stress Level","content":"High stress level - hi migraine!"}
ent6 = {"title":"Mood Level","content":"Bad Mood, bad migraine"}

make_col_entry([ent1,ent2])
make_2col_date_plot(user_data,"migraine_severity","sleep_hours",labels=["Migraine Severity","Hours of Sleep"],episodes=[False,True])
make_col_entry([ent3,ent4])
make_2col_date_plot(user_data,"hydration_level","screen_time",labels=["Hydration Level","Screen Time (h)"])
make_col_entry([ent5,ent6])
make_2col_date_plot(user_data,"stress_level","mood_level",labels=["Stress Level","Mood Level"])

#st.bar_chart(user_data["migraine_severity"])