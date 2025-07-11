from typing import Dict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
}

BASE_URL = "https://nethackwiki.com/wiki"
MONSTERS = f"{BASE_URL}/Monster#Canonical_list_of_monsters"
EXPECTED_MONSTER_FIELDS = [
    "symbol",
    "Difficulty",
    "Attacks",
    "Baselevel",
    "Baseexperience",
    "Speed",
    "BaseAC",
    "Base MR",
    "Alignment",
    "Frequency(bynormal means)",
    "Genocidable",
    "Weight",
    "Nutritional value",
    "Size",
    "Resistances",
    "Resistancesconveyed",
    "facts",
]


class NetHackParser:

    def _get_monsters_urls(self) -> Dict[str, str]:
        monsters = {}
        response = requests.get(MONSTERS, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        top_level_lis = soup.select("ul > li")
        for li in tqdm(top_level_lis):
            nested_ul = li.find("ul")
            if not nested_ul:
                continue  # skip items without a nested list

            for sub_li in nested_ul.find_all("li", recursive=False):
                # get the second <a> tag (the first is usually the image link)
                links = sub_li.find_all("a")
                if len(links) >= 2:
                    monster_tag = links[1]
                    name = monster_tag.get_text(strip=True)
                    monster_url_relative = monster_tag.get("href").split("wiki")[-1]
                    url = BASE_URL + monster_url_relative
                    monsters[name] = url

        return monsters

    def parse_monster_info(monster_url: str) -> dict:
        """Returns a dictionary of attributes for a given monster url."""
        response = requests.get(monster_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        tbody = soup.select("tbody")
        soup = BeautifulSoup(str(tbody), "html.parser")
        rows = soup.find_all("tr")

        monster_data = {}
        facts = []

        for row in rows:
            th = row.find("th")
            tds = row.find_all("td")

            # symbol
            if th and "colspan" in th.attrs:
                name_tag = th.find("span", class_="nhsym")
                if name_tag:
                    monster_data["symbol"] = name_tag.text.strip()

            # Regular stat fields
            elif th and len(tds) == 1:
                key = th.get_text(strip=True).replace(" (by normal means)", "")
                value = tds[0].get_text(strip=True)
                monster_data[key] = value

            # Bullet-point facts
            elif tds and tds[0].find("ul"):
                for li in tds[0].find_all("li"):
                    fact = li.get_text(strip=True)
                    if fact:
                        facts.append(fact)

            # External reference
            elif len(tds) == 2 and "Reference" in tds[0].text:
                monster_data["Reference"] = tds[1].find("a")["href"]

        if facts:
            monster_data["facts"] = facts

        return {k: v for k, v in monster_data.items() if k in EXPECTED_MONSTER_FIELDS}
