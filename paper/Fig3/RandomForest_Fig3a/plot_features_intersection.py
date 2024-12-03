import seaborn as sns
import matplotlib.pyplot as plt
from matminer.featurizers.site.fingerprint import OPSiteFingerprint

names = [i for i in OPSiteFingerprint().feature_labels()]

# Create a Seaborn bar plot
sns.set(style="whitegrid")  # Set the plot style

# You can customize the color palette using sns.color_palette()
# Example: sns.color_palette("Blues_d")

roomT = {
'15': 0.0461,
'1': 0.0435,
'22': 0.0289,
'35': 0.0285,
'10': 0.0259,
'NsUnfilled': 0.0257,
'4': 0.0242,
'3': 0.0230,
'19': 0.0224,
'36': 0.0222,
'Z': 0.0221,
'N Unfilled': 0.0216}

allT = {
        '22': 0.0324,
'10': 0.0304,
'15': 0.0290,
'27': 0.0257,
'5': 0.0256,
'4': 0.0252,
'Z': 0.0238,
'2': 0.0225,
'SpaceGroupNumber': 0.0221,
'N Unfilled': 0.0220,
'36': 0.0219,
'11': 0.0209}

it = set(set(roomT.keys()).intersection(allT.keys()))

s = {}
for i in it:
    s[i] = roomT[i] + allT[i]


s = dict(sorted(s.items(), key=lambda item: item[1], reverse=True))

print(s)
op = []
for i in s:
    if i.isdigit():
        op.append(names[int(i)])
    else:
        op.append(i)

# Create the bar plot
#sns.barplot(x=list(s.keys()), y=list(s.values()), palette="Blues")
sns.barplot(x=op, y=list(s.values()), palette="Blues_r")

# Add labels and title
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)

# Show the plot
plt.show()
