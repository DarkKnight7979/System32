# %%
from PIL import Image, ImageColor

# %%
ImageColor.getcolor('red', 'RGBA')

# %%
ImageColor.getrgb("hsv(0, 100%, 100%)")

# %%
ImageColor.colormap

# %%
for color in ImageColor.colormap:
    print(color, ImageColor.getrgb(color))

# %%
with Image.open("../data/image.png") as img:
    colors = img.getcolors()
    print(colors)

# %%
image = Image.new("RGBA", (150, 150))

# %%
red = ImageColor.getcolor("red", "RGBA")
green = ImageColor.getcolor("green", "RGBA")
blue = ImageColor.getcolor("blue", "RGBA")

color = red

count=0
for y in range(150):
    for x in range(150):
        image.putpixel((x, y), color)
        count += 1
        if count % 10 == 0:
            if color == red:
                color = green
            elif color == green:
                color = blue
            else:
                color = red
            count = 0
    
image.save("../data/image_color.png")

# %%
color_image = Image.open("../data/image.png")
gray_image = color_image.convert("L")
gray_image.save("../data/image_gray.png")

# %%
color_image = Image.open("../data/image.png")
bw_image = color_image.convert("1")
bw_image.save("../data/image_bw.png")

# %%
color_image = Image.open("../data/image.png")
bw_image = color_image.convert("1", dither=0)
bw_image.save("../data/image_bw_no_dither.png")

# %%
color_image = Image.open("../data/image.png")
four_image = color_image.convert("P", palette=Image.ADAPTIVE, colors=4)
four_image.save("../data/image_four.png")

# %%
whitish = (255, 240, 192)
sepia_palette = []

r, g, b = whitish
for i in range(255):
    new_r = r * i // 255
    new_g = g * i // 255
    new_b = b * i // 255
    sepia_palette.extend((new_r, new_g, new_b))


color_image = Image.open("../data/image.png")
gray = color_image.convert("L")
gray.putpalette(sepia_palette)
sepia_image = gray.convert("RGB")

sepia_image.save("../data/image_sepia.png")

# %%



