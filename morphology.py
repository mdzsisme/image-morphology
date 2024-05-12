import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.uint8)

dilation = cv2.dilate(binary_image, kernel)
erosion = cv2.erode(binary_image, kernel)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 12))
plt.subplot(2, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original')
plt.subplot(2, 3, 2)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.subplot(2, 3, 3)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.show()
