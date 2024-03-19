# Define the list of landmark names
landmarks = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Initialize an empty list to store column headers
column_headers = []

# Iterate over each pair of landmarks
for i in range(len(landmarks)):
    for j in range(i + 1, len(landmarks)):
        for coord in ["x", "y", "z"]:
            column_headers.append(f"ratio_{landmarks[i]}_{landmarks[j]}_{coord}")

# Print all column headers
for header in column_headers:
    print(header + ",", end="")
