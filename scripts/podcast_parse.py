import json
from datetime import datetime
from collections import defaultdict
import csv

filenames=["Streaming_History_Audio_2017-2019_0","Streaming_History_Audio_2019-2020_1","Streaming_History_Audio_2020-2021_2","Streaming_History_Audio_2021-2024_3","Streaming_History_Audio_2024_4"]

episode_data = defaultdict(lambda: {"Date": None, "Host": None, "Name": None, "Time": 0})

for filename in filenames:
    with open(f"scripts/data/{filename}.json", "r",encoding="utf-8") as file:
        data=json.load(file)

    for entry in data:
        if entry.get("episode_show_name"):
            episode_name = entry["episode_name"]
            date = datetime.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%SZ").date()
            host = entry["episode_show_name"]
            ms_played = entry["ms_played"]

            # Aggregate the data
            if episode_data[episode_name]["Date"] is None:
                episode_data[episode_name]["Date"] = date
                episode_data[episode_name]["Host"] = host
                episode_data[episode_name]["Name"] = episode_name
            episode_data[episode_name]["Time"] += ms_played

with open(f"scripts/data/streaming_data.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Date", "Host", "Name", "Time"])
    writer.writeheader()

    # Write aggregated data to the CSV
    for episode in episode_data.values():
        writer.writerow(episode)