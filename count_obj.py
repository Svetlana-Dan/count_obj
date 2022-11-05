import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_erosion


#image = np.load('/Users/svetaparilova/Downloads/ps.npy.txt')
image = np.load('ps.npy.txt')

mask1 = np.array([[1,1,1,1], 
                [1,1,1,1], 
                [1,1,0,0],
                [1,1,0,0], 
                [1,1,1,1], 
                [1,1,1,1]])

mask2 = np.array([[1,1,1,1], 
                [1,1,1,1], 
                [0,0,1,1],
                [0,0,1,1], 
                [1,1,1,1], 
                [1,1,1,1]])

mask3 = np.array([[1,1,1,1,1,1],
                [1,1,1,1,1,1], 
                [1,1,1,1,1,1],
                [1,1,1,1,1,1]])

mask4 = np.array([[1,1,1,1,1,1],
                [1,1,1,1,1,1], 
                [1,1,0,0,1,1],
                [1,1,0,0,1,1]])

mask5 = np.array([[1,1,0,0,1,1],
                [1,1,0,0,1,1], 
                [1,1,1,1,1,1],
                [1,1,1,1,1,1]])

counts = []

#1
count1 = binary_erosion(image, mask1)
labeled1 = label(count1)
counts.append(np.max(labeled1))

#2
count2 = binary_erosion(image, mask2)
labeled2 = label(count2)
counts.append(np.max(labeled2))

#3
count3 = binary_erosion(image, mask3)
labeled3 = label(count3)
counts.append(np.max(labeled3))

#4
count4 = binary_erosion(image, mask4)
labeled4 = label(count4)
counts.append(np.max(labeled4)-np.max(labeled3))
count = np.max(labeled4)
print(count)

#5
count5 = binary_erosion(image, mask5)
labeled5 = label(count5)
counts.append(np.max(labeled5)-np.max(labeled3)) 

print(np.array(counts).sum(), counts[0], counts[1], counts[2], counts[3], counts[4])

plt.subplot(121)
plt.imshow(image)

plt.show()

