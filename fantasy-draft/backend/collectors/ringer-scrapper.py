from selenium import webdriver 
from selenium.webdriver.common.by import By
import time
import polars as pl

driver = webdriver.Chrome()

url = 'https://theringer.com/fantasy-football/2025'


driver.get(url)

driver.maximize_window()


try:
    time.sleep(3)  # wait for initial content

    last_height = driver.execute_script("return document.body.scrollHeight")

    data = []   # list of dicts
    seen = set()  # avoid duplicates

    while True:
        # rankings
        ranking_els = driver.find_elements(
            By.XPATH, '//*[@id="main-content"]/div/div[2]/div/div/div/div/div/button/div[1]'
        )
        rankings = [el.get_attribute("textContent") for el in ranking_els]

        # players
        player_els = driver.find_elements(
            By.XPATH, '//*[@id="main-content"]/div/div[2]/div/div/div/div/div/button/div[3]'
        )
        players = [el.get_attribute("textContent") for el in player_els]

        # combine rank-player pairs
        for r, p in zip(rankings, players):
            key = (r, p)
            if key not in seen:
                data.append({"ranking": r, "player": p})
                seen.add(key)
                print(f"Scraped: {r} - {p}")

        # scroll down
        driver.execute_script("window.scrollBy(0, 800);")  # scroll down by 800px
        time.sleep(1)  # short wait for new content

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # convert to DataFrame
    df = pl.DataFrame(data)
    print(df.head())
    print(f"\nTotal scraped rows: {len(df)}")


finally:
    driver.quit()


df.write_csv('data/ringer-ff-rankings.csv')
