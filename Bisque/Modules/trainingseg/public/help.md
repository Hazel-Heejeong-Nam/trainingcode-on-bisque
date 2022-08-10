# Brain Segmentation Training Code 

## I. Input

### Match the number
  
For each subjects, we should have 2 files; One is raw scan and the other is segmented image(ground truth).
If the number of files between **Raw Scans** and **Ground truth images** are different, Bisque will throw an error.

### Naming the files

For the convenience of users, we highly recommend you to put subject's information in your file names like:

> **Raw Scans** : \{  SUBJECT INFO   \}.nii.gz
> **Ground truth images(Segmented)** : Final_\{   SUBJECT INFO   \}.nii.gz

If you did not put **Final\_** in front of **your ground truth** files, module cannot read your files.


## II. Output

When training is done, these are what you are going to get:

> 1. Trained models(.pt)
> 
> 2. Dice Scores(.txt)
> 
> 3. Accuracy(.txt) 
> 
> 4. Output Image (just to notify training is done)
