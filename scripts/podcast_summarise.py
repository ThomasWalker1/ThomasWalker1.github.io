import csv
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

# Load the aggregated data from the CSV file
input_file = "scripts/data/streaming_data.csv"

data = []
with open(input_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row["Time"] = int(row["Time"])
        row["Date"] = datetime.strptime(row["Date"], "%Y-%m-%d").date()
        data.append(row)

# Calculate summary statistics
total_time = sum(entry["Time"] for entry in data)
most_listened_podcast = max(data, key=lambda x: x["Time"])
longest_podcast = max(data, key=lambda x: x["Time"])

# Group data by host to determine trends
host_play_time = defaultdict(int)
monthly_play_time = defaultdict(int)

for entry in data:
    host_play_time[entry["Host"]] += entry["Time"]
    month = entry["Date"].strftime("%Y-%m")
    monthly_play_time[month] += entry["Time"]

most_listened_host = max(host_play_time, key=host_play_time.get)

# Generate summary statistics text file
summary_file = "scripts/data/podcast_summary.txt"
with open(summary_file, mode='w', encoding='utf-8') as file:
    file.write("Podcast Listening Summary:\n")
    file.write(f"Total Time Spent Listening: {total_time / (60*1000):.2f} minutes\n")
    file.write(f"Most Listened Podcast: {most_listened_podcast['Name']} by {most_listened_podcast['Host']} ({most_listened_podcast['Time'] / (60*1000):.2f} minutes)\n")
    file.write(f"Longest Podcast: {longest_podcast['Name']} by {longest_podcast['Host']} ({longest_podcast['Time'] / (60*1000):.2f} minutes)\n")
    file.write(f"Most Listened Host: {most_listened_host} ({host_play_time[most_listened_host] / (60*1000):.2f} minutes)\n")
    file.write("\nHost Listening Breakdown:\n")
    for host, time in sorted(host_play_time.items(),key=lambda item:item[1],reverse=True):
        file.write(f"{host}: {time / (60*1000):.2f} minutes\n")

# Generate Monthly Listening Activity Chart
months = list(monthly_play_time.keys())
monthly_times = [monthly_play_time[month] / (60*1000) for month in months]

plt.figure(figsize=(10, 6))
plt.bar(months, monthly_times, color='skyblue')
plt.title("Monthly Listening Activity")
plt.xlabel("Month")
plt.ylabel("Time (minutes)")
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig("scripts/data/monthly_listening.png",bbox_inches="tight")
plt.close()

# Generate Host Listening Distribution Pie Chart
hosts = list(host_play_time.keys())
host_times = [host_play_time[host] / (60*1000) for host in hosts]

# Sort hosts by play time and keep the top 10
sorted_hosts = sorted(zip(hosts, host_times), key=lambda x: x[1], reverse=True)
top_hosts = sorted_hosts[:10]
remaining_time = sum(x[1] for x in sorted_hosts[10:])

# Prepare data for pie chart
top_host_labels = [host for host, _ in top_hosts] + ["Others"]
top_host_times = [time for _, time in top_hosts] + [remaining_time]

plt.figure(figsize=(8, 8))
plt.pie(top_host_times, labels=top_host_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20c.colors)
plt.title("Host Listening Distribution")
plt.tight_layout()
plt.savefig("scripts/data/host_distribution.png",bbox_inches="tight")
plt.close()

# Generate Recent Podcasts List
recent_podcasts_file = "scripts/data/recent_podcasts.txt"
data.sort(key=lambda x: x["Date"], reverse=True)  # Sort by date, most recent first
with open(recent_podcasts_file, mode='w', encoding='utf-8') as file:
    file.write("Recent Podcasts:\n")
    for entry in data[:25]:  # Get the 10 most recent podcasts
        file.write(f"Date: {entry['Date']}, Host: {entry['Host']}, Episode: {entry['Name']}\n")

print(f"Podcast summary has been written to {summary_file}, with charts saved as images, and recent podcasts saved to {recent_podcasts_file}.")
