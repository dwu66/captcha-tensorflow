from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# Generate a random 4-digit number
digits = [random.randint(0, 9) for i in range(4)]

bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
# Create a new image with a white background
image = Image.new('RGB', (86, 37), color=bg_color)

# Get a drawing context for the image
draw = ImageDraw.Draw(image)

# Add some text to the image
# text = ' '.join(map(str, digits))
font = ImageFont.truetype('arial.ttf', size=16)
# draw.text((12, 7), text, font=font, fill=(0, 0, 0))

x = 16
y = 8
for digit in digits:
    # Generate a random color for the digit
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw.text((x, y), str(digit), font=font, fill=color)
    x += 14  # move the next digit over by the width of one digit

# Apply a blur filter to the image
blur_radius = 0.5
image = image.filter(ImageFilter.GaussianBlur(blur_radius))

# # Add some noise dots to the image
for i in range(200):
    x = random.randint(0, 199)
    y = random.randint(0, 99)
    draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

blur_radius = 0.5
image = image.filter(ImageFilter.GaussianBlur(blur_radius))

# Save the image to a file
image.save('4_digit_image13.png')
