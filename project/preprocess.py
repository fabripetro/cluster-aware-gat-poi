import pandas as pd
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the TKY dataset
tky_data_path = "dataset_TSMC2014_TKY.txt"

df_tky = pd.read_csv(tky_data_path, sep="\t", header=None, names=[
    "user_id", "venue_id", "venue_category_id", "venue_category_name",
    "latitude", "longitude", "timezone_offset", "timestamp"
], encoding="ISO-8859-1")

# Converting the column timestamp in datetime on TKY dataset
df_tky["timestamp"] = pd.to_datetime(df_tky["timestamp"], utc=True)

# Apllying .timestamp() to obtain the nuemeric value of Unix on NYC dataset
df_tky["timestamp"] = df_tky["timestamp"].apply(lambda x: x.timestamp())

# Mapping for TKY's users
user_mapping1 = {id: idx for idx, id in enumerate(df_tky['user_id'].unique(), start=1083)}

# Mapping for TKY's POI
poi_mapping1 = {id: idx for idx, id in enumerate(df_tky['venue_id'].unique(), start=38333)}

# Mapping for TKY's POI categories
category_mapping1 = {id: idx for idx, id in enumerate(df_tky['venue_category_id'].unique(), start=400)}

# Substitute the ID's into the DataFrame
df_tky['user_id'] = df_tky['user_id'].map(user_mapping1)
df_tky['venue_id'] = df_tky['venue_id'].map(poi_mapping1)
df_tky['venue_category_id'] = df_tky['venue_category_id'].map(category_mapping1)

# Save the modified dataset
#df_tky.to_csv("tky.txt", sep="\t", index=False, encoding="ISO-8859-1")


# -------------------------------------------------------------------------------------------------------------------- #


df_tky["user_id"] = df_tky["user_id"].astype(int)
df_tky["venue_id"] = df_tky["venue_id"].astype(int)
df_tky["venue_category_id"] = df_tky["venue_category_id"].astype(int)
df_tky["venue_category_name"] = df_tky["venue_category_name"].astype(str)
df_tky["latitude"] = pd.to_numeric(df_tky["latitude"], errors='coerce')
df_tky["longitude"] = pd.to_numeric(df_tky["longitude"], errors='coerce')
df_tky["timezone_offset"] = pd.to_numeric(df_tky["timezone_offset"], errors='coerce')
df_tky["timestamp"] = pd.to_numeric(df_tky["timestamp"].astype(int), errors='coerce')

max_user = df_tky['user_id'].nunique()

max_poi_count = df_tky['venue_id'].nunique()

max_poi_cat_count = df_tky['venue_category_id'].nunique()


# Group by user_id
user_checkins = df_tky.groupby("user_id")

# List to store the check-in history
check_in_data = []
check_in_data_cut = []
# min and max time
min_time = min(df_tky['timestamp'])
max_time = max(df_tky['timestamp'])

# Vectorize latitude and longitude for all users
latitudes = torch.tensor(df_tky["latitude"].values, dtype=torch.float64, device=device)
longitudes = torch.tensor(df_tky["longitude"].values, dtype=torch.float64, device=device)

# Compute the coentroid of the dataset
lat_centroid = df_tky["latitude"].mean()
lon_centroid = df_tky["longitude"].mean()

# Compute the distance from the centroid
distances = torch.sqrt((latitudes - lat_centroid)**2 + (longitudes - lon_centroid)**2)

# Set to store unique venue_id values
unique_venue_ids = set()
unique_venue_ids_cut = set()

# Dictionary to store the distance of each POI (Point of Interest) from the centroid
venue_distances = {}
venue_cat = {}
venue_distances_cut = {}
venue_distances_cut_coord = {}
venue_cat_cut = {}

# Dictionary that store, for each POI's id, all the categories that are associated with it
# (can be happen that foe the same poi, we have different categories)
poi_list_categories = {}

# Iterate over user groups
for user_id, group in user_checkins:

    # Create a check-in list for each user
    check_ins = []
    last = []
    user_indices = group.index
    cut_idx = user_indices[-100]

    for idx in user_indices:

        row = df_tky.loc[idx]

        usr_id = int(row["user_id"])
        venue_id = int(row["venue_id"])
        cat_id = int(row["venue_category_id"])
        lat = row["latitude"]
        lon = row["longitude"]

        # Compute the normalized time
        normalized_time = (int(row["timestamp"]) - min_time) / (max_time - min_time)

        # Taking the distance
        distance = distances[idx]

        unique_venue_ids.add(venue_id)
        if venue_id not in venue_distances:
            venue_distances[venue_id] = distance
        if venue_id not in venue_cat:
            venue_cat[venue_id] = cat_id
            poi_list_categories[venue_id] = [cat_id]
        else:
            tmp_list_cat = poi_list_categories[venue_id]
            if cat_id not in tmp_list_cat:
                poi_list_categories[venue_id].append(cat_id)

        # Create the vector for check-in
        check_in = [usr_id, venue_id, venue_cat[venue_id], venue_distances[venue_id], normalized_time]
        check_ins.append(check_in)

        if cut_idx <= idx:
            unique_venue_ids_cut.add(venue_id)
            if venue_id not in venue_distances_cut:
                venue_distances_cut[venue_id] = distance
            if venue_id not in venue_distances_cut_coord:
                venue_distances_cut_coord[venue_id] = (lat, lon)
            if venue_id not in venue_cat_cut:
                venue_cat_cut[venue_id] = cat_id
            last.append(check_in)

    # Add the check-in list for this user
    check_in_data.append(check_ins)
    check_in_data_cut.append(last)

max_checkins = max(len(user_checkins) for user_checkins in check_in_data)
min_checkins = min(len(user_checkins) for user_checkins in check_in_data)

# Adding padding equal to zero
for i in range(len(check_in_data)):
    while len(check_in_data[i]) < max_checkins:
        check_in_data[i].append([0, 0, 0, 0.0, 0.0])


# Now create a list of [venue_id, distance] pairs
poi_with_distances = [[venue_id, venue_distances_cut[venue_id]] for venue_id in unique_venue_ids_cut]
poi_with_coord = [[venue_id, venue_distances_cut_coord[venue_id][0], venue_distances_cut_coord[venue_id][1]] for venue_id in unique_venue_ids_cut]
poi_with_cat = [[venue_id, venue_cat_cut[venue_id]] for venue_id in unique_venue_ids_cut]


check_in_tuples_history = torch.tensor(check_in_data, dtype=torch.float64, device=device)  # UNCUT
torch.save(check_in_tuples_history, "check_in_tuples_history")


choose_K(dataset=poi_with_coord, max_K=31)


ids, clusters = K_Means(dataset=poi_with_coord, n_clusters=6)
clusters = clusters.tolist()
ids_clusters = dict(zip(ids, clusters))


for i in range(len(check_in_data_cut)):
    for j in range(len(check_in_data_cut[i])):
        check_in_data_cut[i][j].append(ids_clusters[check_in_data_cut[i][j][1]])


# Convert the list into a PyTorch tensor
tensor_poi_coord = torch.tensor(poi_with_coord, dtype=torch.float64, device=device)    # POI, LAT, LONG (CUT 100)
tensor_poi_locations = torch.tensor(poi_with_distances, dtype=torch.float64, device=device)    # POI, POS (CUT 100)
tensor_poi_categories = torch.tensor(poi_with_cat, dtype=torch.float64, device=device)     # POI, CAT (CUT 100)

check_in_tuples_history_cut = torch.tensor(check_in_data_cut, dtype=torch.float64, device=device)  # CUT 100


torch.save(tensor_poi_coord, "tensor_poi_coord")
torch.save(tensor_poi_locations, "tensor_poi_locations")
torch.save(tensor_poi_categories, "tensor_poi_categories")
torch.save(check_in_tuples_history_cut, "check_in_tuples_history_cut")


train_dataset, validation_dataset, test_dataset = train_val_test_split_tensor(check_in_tuples_history_cut)
torch.save(train_dataset, "train_dataset")
torch.save(validation_dataset, "validation_dataset")
torch.save(test_dataset, "test_dataset")


# -------------------------------------------------------------------------------------------------------------------- #


num_checkin_u = {}
for user_id in range(check_in_tuples_history.shape[0]):

    id_u = int(check_in_tuples_history[user_id][0][0])
    num_checkin = (torch.any(check_in_tuples_history[user_id] != 0, dim=1)).sum().item()
    num_checkin_u[id_u] = num_checkin


# Extract the number of check-ins into a list
num_checkin = list(num_checkin_u.values())

# Compute the mean and the standard deviation
mean_checkin = np.mean(num_checkin)
std_checkin = np.std(num_checkin)

max_val = max(num_checkin)
min_val = min(num_checkin)
num_occ = len(list(filter(lambda x: x > 110, num_checkin)))

# Probability plot (prob that one user has at least x checkin)
num_checkin_arr = np.array(num_checkin)
x_vals = np.linspace(0, max_val, 1000)
probs = [np.mean(num_checkin_arr >= x) for x in x_vals]
# Desired probability
p = 0.90
x = np.percentile(num_checkin_arr, 100 * (1 - p))

# Create a histogram showing the distribution of check-ins
plt.figure(figsize=(10, 6))
sns.histplot(num_checkin, kde=False, color='blue', bins=100, stat='count')
counts, bin_edges = np.histogram(num_checkin, bins=100)
print(bin_edges)

# Calculate the bin width
bin_width = np.mean(np.diff(bin_edges))

# Add a line for the normal (Gaussian) distribution
xmin_plt1, xmax_plt1 = plt.xlim()
x_plt1 = np.linspace(xmin_plt1, xmax_plt1, 1000)

p_plt1 = (1 / (std_checkin * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plt1 - mean_checkin) / std_checkin) ** 2)
# Scale the distribution to match the counts: multiply it by the number of data points and the bin width
p_plt1_scaled = p_plt1 * len(num_checkin) * bin_width

plt.plot(x_plt1, p_plt1_scaled, 'k', linewidth=2, label=f"Normal distribution (scaled, μ={mean_checkin:.2f}, σ={std_checkin:.2f})")

xticks_plt1 = np.arange(0, max_val + 200, 200)
x_peak_plt1 = x_plt1[np.argmax(p_plt1_scaled)]
plt.axvline(x_peak_plt1, color='red', linestyle='--', label=f'# Maximum of the Gaussian: {x_peak_plt1:.2f}')
plt.xticks(xticks_plt1, fontsize=12)

plt.title("Distribution of the Number of Check-ins")
plt.xlabel("Number of Check-ins")
plt.ylabel("Number of users")
plt.legend()
plt.savefig("Distribution of the Number of Check-ins.png", dpi=300, bbox_inches='tight')
#plt.show()

# -----------------------------
# Probability plot (prob that one user has at least x checkin)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, probs, color='darkred')
plt.title("P(user has ≥ x check-ins)")
plt.xlabel("Number of check-ins (x)")
plt.ylabel("Probability")
plt.grid(True)
plt.savefig("Probability Plot.png", dpi=300, bbox_inches='tight')
#plt.show()

print("\nCheck-in Statistics:")
print("----------------------------")
print(f"Total number of users: {len(num_checkin):,}")
print(f"Mean number of check-ins: {mean_checkin:.2f}")
print(f"Standard deviation of check-ins: {std_checkin:.2f}")
print(f"Maximum check-in value: {max_val}")
print(f"Minimum check-in value: {min_val}")
print(f"Number of users with more than 110 check-ins: {num_occ}")
print(f"At least {p*100:.0f}% of users have ≥ {x:.0f} check-ins")
print("----------------------------\n")
