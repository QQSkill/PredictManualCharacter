import pygame
from pygame.locals import *
from pygame import image
from pygame.locals import * 
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model('classifier.h5')

run = True
is_writing = False
WINDOWSIZE_X = 640
WINDOWSIZE_Y = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED  = (255, 0, 0)

IMAGE_SAVE = False
PREDICT = True
LABELS = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

pygame.init()

FONT = pygame.font.SysFont('Futura', 30)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZE_X, WINDOWSIZE_Y))

pygame.display.set_caption('DIGIT PREDICT GAME')

BOUNDARY = 5
number_xcord = []
number_ycord = []

while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEMOTION and is_writing == True:
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x, y), 4, 0)
            number_xcord.append(x)
            number_ycord.append(y)
        if event.type == pygame.MOUSEBUTTONDOWN:
            is_writing = True
        if event.type == pygame.MOUSEBUTTONUP:
            IMAGE_SAVE = True
            is_writing = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARY, 0), min(WINDOWSIZE_X, number_xcord[-1] + BOUNDARY)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARY, 0), min(number_ycord[-1] + BOUNDARY, WINDOWSIZE_X)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            img = None
            if IMAGE_SAVE:
                cv2.imwrite('digit.png', img_arr)
                img = pygame.image.load('digit.png').convert_alpha()
                IMAGE_SAVE = False
                #image_count += 1
            
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = img.get_rect()
                textRectObj.x = rect_min_x
                textRectObj.y = rect_min_y
                #textRectObj = testing.get_rect()
                #textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y
                pygame.draw.rect(DISPLAYSURF, RED, textRectObj, 2)
                DISPLAYSURF.blit(textSurface, (rect_min_x, rect_max_y))
            if event.type == KEYDOWN:
                 if event.unicode == 'n':
                     DISPLAYSURF.fill(BLACK)
        pygame.display.update()