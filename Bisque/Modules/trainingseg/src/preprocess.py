import nibabel as nib
import numpy as np
import os
from collections import defaultdict
import random


def getPatch(image, segmentation, i, j, k):

    sample=image[i-16:i+16+1,j-16:j+16+1,k-1:k+1+1]
    annotation=segmentation[i-16:i+16+1,j-16:j+16+1,k]
    center=segmentation[i-1:i+1+1,j-1:j+1+1,k]
    
    return sample, annotation, center

def readImage(image, annot):
    
    image_arr=nib.load(image).get_fdata()
    annot_arr=nib.load(annot).get_fdata()
    
    return image_arr, annot_arr

def getPatchMulticlass(group, imageName, annotName, imageBase, annotBase, Base ,classNum=7):   
    
    classes=[0]*7
    multiclasses=[0]*7
    count=0
    multiclassData=defaultdict(list)
    data=defaultdict(list)
    image, annotation=readImage(os.path.join(imageBase, imageName), os.path.join(annotBase, annotName))
    prefix=imageName.split('.')[0]
    x,y,z=annotation.shape


    with open(os.path.join(Base,'image_shape.txt'),"a+") as file:
        file.write('{},{},{},{}\n'.format(prefix,x,y,z))

    for k in range(1, z-1, 1):
        for i in range(17, x-17, 2):
            for j in range(17, y-17, 2):

                if annotation[i,j,k]!=0: 
                    
                    sample, label, center=getPatch(image, annotation, i, j, k)
                    if center.any() and np.unique(center).size>=2:
                        multiclasses[int(annotation[i,j,k])]+=1
                        multiclassData[int(annotation[i,j,k])].append((i,j,k))
                    
                    elif center.all() and np.unique(label).size>=2:
                        classes[int(annotation[i,j,k])]+=1
                        data[int(annotation[i,j,k])].append((i,j,k))
                        
                    
    print(multiclasses, end=' ')
    print(classes)
 
    for cla in range(1, len(classes)):
        
        numCurClass=min(multiclasses[cla], 10*multiclasses[6])
        positions=multiclassData[cla]
        random.shuffle(positions)
        for pos in range(numCurClass):

            i,j,k=positions[pos]

            sample, _, center=getPatch(image, annotation, i, j, k)
            name1='image{}_{}.npy'.format(prefix, count)
            name2='label{}_{}.npy'.format(prefix, count)
            with open(os.path.join(Base,'Scans_patch_mixed/')+name1, 'wb') as f:
                np.save(f, sample)

            with open(os.path.join(Base,'Segmentation_patch_mixed/')+name2, 'wb') as f:
                np.save(f, center)


            with open(os.path.join(Base,'{}_positions_mixed.txt'.format(group)),"a+") as file:
                file.write('{},{},{},{},{},{}\n'.format(prefix, name1, name2,i,j,k))

            count+=1 

    for cla in range(1, len(classes)):
        numCurClass=min(classes[cla], 4*multiclasses[6])

        positions=data[cla]
        random.shuffle(positions)
        for pos in range(numCurClass):

            i,j,k=positions[pos]

            sample, _, center=getPatch(image, annotation, i, j, k)
            name1='image{}_{}.npy'.format(prefix, count)
            name2='label{}_{}.npy'.format(prefix, count)

            with open(os.path.join(Base,'Scans_patch_mixed/')+name1, 'wb') as f:
                np.save(f, sample)

            with open(os.path.join(Base,'Segmentation_patch_mixed/')+name2, 'wb') as f:
                np.save(f, center)

            with open(os.path.join(Base,'{}_positions_mixed.txt'.format(group)),"a+") as file:
                file.write('{},{},{},{},{},{}\n'.format(prefix, name1, name2,i,j,k))

            count+=1 

def getPatchRand(group, imageName, annotName, imageBase, annotBase, Base, classNum=7):   
    
    classes=[0]*7
    multiclasses=[0]*7
    count=0
    multiclassData=defaultdict(list)
    data=defaultdict(list)
    image, annotation=readImage(os.path.join(imageBase, imageName), os.path.join(annotBase, annotName))
    prefix=imageName.split('.')[0]
    x,y,z=annotation.shape
    with open(os.path.join(Base,'image_shape.txt'),"a+") as file:
        file.write('{},{},{},{}\n'.format(prefix,x,y,z))

    for k in range(1, z-1, 1):
        for i in range(17, x-17, 3):
            for j in range(17, y-17, 3):

                if annotation[i,j,k]!=0 and image[i,j,k]>-200 and image[i,j,k]<200: 
                    
                    sample, label, center=getPatch(image, annotation, i, j, k)
                    if label.all():
                        classes[int(annotation[i,j,k])]+=1
                        data[int(annotation[i,j,k])].append((i,j,k))
                        
    num=min(classes[1:])                    
    for cla in range(1, len(classes)):
        
        positions=data[cla]
        random.shuffle(positions)
        for pos in range(num):

            i,j,k=positions[pos]

            sample, _, center=getPatch(image, annotation, i, j, k)
            name1='image{}_{}.npy'.format(prefix, count)
            name2='label{}_{}.npy'.format(prefix, count)

            with open(os.path.join(Base,'Scan_patch_test/')+name1, 'wb') as f:
                np.save(f, sample)

            with open(os.path.join(Base,'Segmentation_patch_test/')+name2, 'wb') as f:
                np.save(f, center)

            with open(os.path.join(Base,'{}_positions_rand.txt'.format(group)),"a+") as file:
                file.write('{},{},{},{},{},{}\n'.format(prefix, name1, name2,i,j,k))

            count+=1 

