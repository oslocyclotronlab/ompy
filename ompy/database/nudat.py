import mechanicalsoup
import pandas as pd
from .nucleus import Nucleus, LevelScheme, GammaScheme


def get_nucleus_df(nucleus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    browser = mechanicalsoup.StatefulBrowser(user_agent="MechanicalSoup")
    browser.open("https://www.nndc.bnl.gov/nudat3/indx_adopted.jsp")
    browser.select_form("form")
    form = browser.get_current_form()
    form["nuc"] = nucleus
    form['out'] = 'file'

    response = browser.submit_selected()
    lines = response.text.split('\n')
    lines = lines[4:-5]

    # We get back two tables. They are separated by a line with ' '.
    table1: list[list[str]] = []
    table2: list[list[str]] = []
    table = table1
    for line in lines:
        if line == ' ':
            table = table2
            continue
        table.append([l.strip() for l in line.split('\t')])

    levels = pd.DataFrame(table1[1:])
    levels.columns = table1[0]
    levels['Energy'] = levels['Energy'].astype(float)

    gammas = pd.DataFrame(table2[1:])
    gammas.columns = table2[0]
    gammas['Energy'] = gammas['Energy'].astype(float)
    gammas['E Gamma'] = gammas['E Gamma'].astype(float)

    return levels, gammas

def get_nucleus(nucleus: str) -> Nucleus:
    levels, gammas = get_nucleus_df(nucleus)
    nuc = Nucleus(nucleus)
    levels_ = LevelScheme()
    gammas_ = GammaScheme()
    for i, level in levels.iterrows():
        levels_.add(level['Energy'], level['JPi'])
    for i, gamma in gammas.iterrows():
        gammas_.add(gamma['Energy'], gamma['E Gamma'], gamma['JPi'], gamma['T1/2 (txt)'])
    nuc.levels = levels_
    nuc.gammas = gammas_
    return nuc


if __name__ == '__main__':
    nucleus = get_nucleus("12C")
    print(nucleus)
