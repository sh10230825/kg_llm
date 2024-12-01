import os
import requests
from lxml import html
import pandas as pd
import time

# output path
output_dir = os.path.join(os.path.dirname(__file__), "../dataset")
os.makedirs(output_dir, exist_ok=True)  # ensure path exist

# URL
base_url = "https://www.cmdhi.mohw.gov.tw/Interactions/Detail?id={}"

# safe extract func.
def safe_extract(tree, xpath, multiple=False):
    try:
        result = tree.xpath(xpath)
        if not result:
            return None
        if multiple:
            return result
        else:
            return result[0].text.strip() if result[0].text else None
    except Exception:
        return None

# fetch ID data
def fetch_data(id):
    url = base_url.format(id)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"ID {id}: HTTP {response.status_code}")
            return None, [], []
        response.encoding = 'utf-8'
        tree = html.fromstring(response.text)
        
        # fetch main
        data = {
            "ID": id,
            "中藥中文名": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[1]/td/div/table/tbody/tr[2]/td/h4/a"),
            "中藥拉丁學名": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[1]/td/div/table/tbody/tr[3]/td/h4"),
            "中藥英文名": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[1]/td/div/table/tbody/tr[4]/td/h4"),
            "中藥基原": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[1]/td/div/table/tbody/tr[5]/td/h4"),
            "西藥學名": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[2]/td/div/table/tbody/tr[2]/td/h4"),
            "建議處理方式": safe_extract(tree, "/html/body/div[2]/div/div[2]/div[2]/table[1]/tbody/tr[3]/td/div/table/tbody/tr[2]/td/h4"),
        }
        
        # fetch research
        research_rows = tree.xpath("/html/body/div[2]/div/div[2]/div[2]/div/table/tbody/tr/td/table/tbody[2]/tr")
        research_data = []
        for row in research_rows:
            cols = row.xpath("td")
            research_data.append({
                "ID": id,
                "項次": cols[0].text.strip() if len(cols) > 0 and cols[0].text else None,
                "研究設計": cols[1].text.strip() if len(cols) > 1 and cols[1].text else None,
                "可能交互作用機轉": cols[2].text.strip() if len(cols) > 2 and cols[2].text else None,
                "可能交互作用結果": cols[3].text.strip() if len(cols) > 3 and cols[3].text else None,
                "參考文獻": cols[4].text.strip() if len(cols) > 4 and cols[4].text else None,
            })
        
        # fetch reference
        reference_rows = tree.xpath("/html/body/div[2]/div/div[2]/div[2]/table[2]/tbody/tr")
        reference_data = []
        for row in reference_rows:
            cols = row.xpath("td")
            reference_data.append({
                "ID": id,
                "項次": cols[0].text.strip() if len(cols) > 0 and cols[0].text else None,
                "篇名": cols[1].text.strip() if len(cols) > 1 and cols[1].text else None,
                "連結": cols[2].xpath("a/@href")[0].strip() if len(cols) > 2 and cols[2].xpath("a/@href") else None,
            })
        
        return data, research_data, reference_data
    except Exception as e:
        print(f"Error fetching ID {id}: {e}")
        return None, [], []

# main：fetch
main_data = []
research_data = []
reference_data = []

start_id = 1
end_id = 3085

for id in range(start_id, end_id + 1):
    print(f"Fetching ID {id}...")
    main, research, references = fetch_data(id)
    if main:
        main_data.append(main)
    if research:
        research_data.extend(research)
    if references:
        reference_data.extend(references)
    time.sleep(1)  # delay 1s

# convert to DataFrame
df_main = pd.DataFrame(main_data)
df_research = pd.DataFrame(research_data)
df_references = pd.DataFrame(reference_data)

# merge data
merged_data = df_main

# merge research_data
if not df_research.empty:
    df_research_grouped = df_research.groupby("ID").agg(list).reset_index()
    merged_data = pd.merge(merged_data, df_research_grouped, on="ID", how="left")

# merge reference_data
if not df_references.empty:
    df_references_grouped = df_references.groupby("ID").agg(list).reset_index()
    merged_data = pd.merge(merged_data, df_references_grouped, on="ID", how="left")

# delete null rows
merged_data = merged_data.dropna(how="all", subset=merged_data.columns.difference(["ID"]))

# output to CSV
output_file_csv = os.path.join(output_dir, "medical_dataset.csv")
merged_data.to_csv(output_file_csv, index=False)

# output to JSON
output_file_json = os.path.join(output_dir, "medical_dataset.json")
merged_data.to_json(output_file_json, orient="records", force_ascii=False, indent=4)

print(f"Fetch completed, output file CSV: {output_file_csv} and JSON: {output_file_json}")
