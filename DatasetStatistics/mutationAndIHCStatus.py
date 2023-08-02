import pandas as pd
import os
import matplotlib.pyplot as plt

dataPath = r"D:\BAP1\Data"
labels = pd.read_csv(os.path.join(dataPath, "BAP1DataCuration.csv"))

labels["IHC BAP1 Status"].loc[labels["IHC BAP1 Status"] == "Lost"] = "Loss"

table = pd.crosstab(labels["Somatic BAP1 mutation status"], labels["IHC BAP1 Status"])

fig, ax = plt.subplots(figsize = (8, 8))
plt.title("BAP1 IHC and Somatic Mutation Status")

bar = table.plot.bar(stacked = True, ax = ax,
               xlabel = "Somatic Mutation Status",
               ylabel = "Count")

plt.legend(["Loss", "Not Done", "Retained"], title = "IHC Status")

plt.show()
