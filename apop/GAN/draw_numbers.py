import os

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


if __name__ == '__main__':
  BLACK = 0  # (0, 0, 0)
  WHITE = 255  # (255, 255, 255)

  for i in range(10):
    # img = Image.new('RGB', (500, 500), (255, 255, 255))
    img = Image.new('L', (28, 28), BLACK)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("open-sans/OpenSans-Regular.ttf", 26)
    # draw.text((150, -30), str(i), (0, 0, 0), font=font)
    draw.text((6, -4), str(i), WHITE, font=font)
    # img.show()

    if not os.path.isdir('imgs/'):
      os.makedirs('imgs/')
    img.save('imgs/digit_number_img_'+str(i)+'.png')