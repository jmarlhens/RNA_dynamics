from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt


def hex_to_rgb(hex):
    h = hex.lstrip("#")
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


CMAP_COLORS = ["#FFFFFF", "#FF5500", "#B3003C"]
cmap_colors = CMAP_COLORS
cmap_colors = [hex_to_rgb(hex_val) for hex_val in cmap_colors]
cmap_colors = np.array(cmap_colors)
cmap = LinearSegmentedColormap.from_list("my_cmap", cmap_colors / 255)

# cmap from yellow to red
cmap_colors = ["#FFFF00", "#FF0000"]

cmap_colors = [hex_to_rgb(hex_val) for hex_val in cmap_colors]
cmap_colors = np.array(cmap_colors)
cmap = LinearSegmentedColormap.from_list("my_cmap", cmap_colors / 255)

# print color map
fig, ax = plt.subplots()
cmap = LinearSegmentedColormap.from_list("my_cmap", cmap_colors / 255)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
cbar.set_label("Color map")
plt.show()


# load Color scheme.xslx

# # first col is the id, forst row is the column names
# color_scheme = pd.read_excel("Color scheme.xlsx", index_col=0, header=0)
# dark_blue = ast.literal_eval(color_scheme.loc["Dark blue", "Python color code"][1])
# print(dark_blue)
#
# fig, ax = plt.subplots()
# ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color=dark_blue))
# plt.show()
