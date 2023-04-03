from captcha.image import ImageCaptcha
import random
from PIL import ImageFont

# Customized digit format
digits = [random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) for i in range(4)]

# with open("./RobotoMono-Regular.ttf", "rb") as f:
#     font = ImageFont.truetype(f, size=12)
# font = ImageFont.truetype("./RobotoMono-Regular.ttf", size=12)
# font = ImageFont.load("./RobotoMono-Regular.ttf")
# Generate the captcha image
# image = ImageCaptcha(width=86, height=37, fonts=["./Arial.ttf"], font_sizes=[20, 20, 20, 20])

image = ImageCaptcha(width=86, height=37, font_sizes=[24, 24, 24, 24], noise_dots=0, noise_curve=0, noise_lines=0,
draw_lines=False, draw_points=False)
captcha = ''.join(digits)
data = image.generate(captcha)

# Save the image to file
with open(captcha + '.png', 'wb') as f:
    f.write(data.read())
