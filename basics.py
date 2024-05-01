# %%
from PIL import Image

# %%
img = Image.open("../data/image.png")
img.show()

# %%
width, height = img.size
print("width: ", width)
print("heigh: ", height)

# %%
img.filename

# %%
img.format

# %%
img.format_description

# %%
img.save("../data/image.jpg", quality=95)

# %%



