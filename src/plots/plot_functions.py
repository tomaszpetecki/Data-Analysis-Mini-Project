import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.api.types import is_numeric_dtype

from src.config import VARS, USER_ID_COLUMN, DATE_COLUMN, PROCESSED_DATA_DIR
from src.utils.functions import find_episodes
from src.utils.functions import bin_data

#FUNCTION for plotting ONE feature on an Axis
def plot_feature(df:pd.DataFrame, feature:str, user_id:int|None,axes: plt.Axes,episodes:bool|list =True):
    "user_id is only for the title, dataframe is supposed to contain data from one user"
    if isinstance(episodes,bool) and episodes:
        ep = find_episodes(df,target="migraine_tomorrow")
    elif isinstance(episodes,list):
        ep = episodes.copy()
    else:
        ep = []
        
    date_values = df[DATE_COLUMN].values
    feature_values = df[feature].values

    axes.plot(date_values, feature_values, label=feature, color='blue')
    axes.set_title(f"{feature} {f'for User {user_id}' if user_id else ''}")
    axes.set_ylabel(feature)
    axes.grid(True)
    
    for start_x, end_x in ep:
        if start_x != end_x:
            axes.axvspan(start_x,end_x,color="red",alpha = 0.3)
        else:
            axes.axvspan(start_x-pd.Timedelta(days=0.5),end_x+pd.Timedelta(days=0.5),color="red",alpha = 0.3)

#FUNCTION for plotting MANY features in one plot
def plot_features(df:pd.DataFrame, user_id, features=VARS,episodes:bool|list = True):
    user_data = df[df[USER_ID_COLUMN] == user_id]

    fig, axes = plt.subplots(len(features), 1, figsize=(10, 12), sharex=True)

    if len(features) == 1:
        axes = [axes]  # To handle the case where we have a single feature plot
    # Plot each feature in a separate subplot
    for i, feature in enumerate(features):
        plot_feature(user_data,feature,user_id,axes[i],episodes=episodes)
    axes[-1].set_xlabel('Date') 

    plt.tight_layout()
    return fig, axes

def plot_pair_plot(df: pd.DataFrame, features: list, hue: str = None,jitter:bool = True):
    def add_jitter(df1, jitter_strength=0.05):
        true_features = [i for i in VARS if i not in ["migraine_severity","migraine_occurrence"]]
        df_jittered = df1.copy()
        for column in true_features:
            df_jittered[column] = df1[column] + jitter_strength * np.random.randn(len(df1))
        return df_jittered
    
    df = df.copy
    if jitter:
        df = add_jitter(df)
    
    sns.set_theme(style="ticks")
    pair_plot = sns.pairplot(df[features], hue=hue,plot_kws={"s":10})  # You can pass the hue column if needed (e.g., a category like 'migraine_tomorrow')
    pair_plot.figure.set_size_inches(10, 10)  # Customize the size of the pair plot
    plt.suptitle("Pair Plot of Features", size=16)
    plt.subplots_adjust(top=0.95)  # Adjust the title position
    plt.show()
    return pair_plot

#Function that returns heatmaps for two features, the heat is the averge of binary classification
def plot_binned_probability_heatmap(df:pd.DataFrame,x:str,y:str,target:str="migraine_tomorrow",ax: plt.Axes|None = None, title=None) -> tuple[plt.Figure,plt.Axes]:
    "This function plot a heatmap of mean(target) over x and y"
    "It will decide automatically whether to bin the continuous values or not"

    #If we give an ax then we draw on ax, otherwise we create and axis
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.figure

    x_bins = bin_data(df[x])
    y_bins = bin_data(df[y])

    df = df.copy()
    df["x_bins"] = x_bins[0] if len(x_bins) == 2 else x_bins
    df["y_bins"] = y_bins[0] if len(y_bins) == 2 else y_bins

    colorbar = ax == fig.axes[-1] # when lats, add colorbar

    pivot = df.pivot_table(values=target,index="y_bins",columns="x_bins",aggfunc="mean",observed=False)
    sns.heatmap(pivot,ax=ax,annot=True,fmt=".2f",cmap="YlOrRd",cbar_kws={"label":f"P({target})"},cbar=colorbar)

    label_settings = {"fontsize":14}
    ax.set_xlabel(x,label_settings)
    ax.set_ylabel(y,label_settings)

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    if title is None:
        ax.set_title(f"{target} probability by {x} and {y}")
    else:
        pass

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    ax.margins(x=0.5, y=0.5)

    return fig, ax

if __name__ == "__main__":
    from src.data.preprocess import preprocess
    from src.data.preprocess import clean_data
    from src.config import PROJECT_ROOT
    FIGURES_DIR = (PROJECT_ROOT / "src" / "reports" / "figures")

    df = pd.read_csv(PROCESSED_DATA_DIR / "clean_data.csv")
    df = clean_data(df)

    if False:
        fig, axes = plot_features(df,4,episodes=True)
        plt.show()
        from src.config import PROJECT_ROOT
        fig.savefig(FIGURES_DIR / "raw_data_plot1.svg", format="svg")
    
    if False:
        plot_pair_plot(add_jitter(df),[i for i in VARS if i not in ["migraine_severity","migraine_occurrence"]]+["migraine_tomorrow"],hue="migraine_tomorrow")
    
    #print(df["screen_time"].min())
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    plot_binned_probability_heatmap(df,x="sleep_hours",y="screen_time",target="migraine_tomorrow", title= " ", ax=axes[0])
    plot_binned_probability_heatmap(df,x="hydration_level",y="mood_level",target="migraine_tomorrow", title=" ", ax=axes[1])
    fig.suptitle("Probability of a Migraine Tomorrow – Heatmap", fontweight="bold")
    
    
    plt.show()
    fig.savefig(FIGURES_DIR / "eda_heatmap.svg", format="svg")