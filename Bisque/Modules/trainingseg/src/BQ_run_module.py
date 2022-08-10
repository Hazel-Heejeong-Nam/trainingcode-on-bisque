import os
import math
from preprocess import *
from TrainModel import *
from torch.utils.data import DataLoader

def run_module(input_path_dict,output_folder_path):
    """
    This function should load input resources from input_path_dict, do any pre-processing steps, run the algorithm,
    save all outputs to output_folder_path, AND return the outputs_path_dict.
    """
    ###Get input file paths from dictionary###
    scans_path = input_path_dict['Raw CT Images']
    seg_path = input_path_dict['Ground Truth Images']


    ###Load data###
    segFiles=[]
    scanFiles = []
    segFiles+=[d for d in os.listdir(seg_path) if 'Final' in d and '.nii.gz' in d] 
    scanFiles += [d for d in os.listdir(scans_path) if '.nii.gz' in d]
    count=len(segFiles)

    if len(scanFiles) != count :
        print('Amount of CT Scans and Ground truth are different')
        return len(scanFiles), count 
    print('Total number of file :',count)


    ###Preprocessing###
    print('Start Preprocessing')
    DataFolders=['train','val','test']

    trainnum = math.floor(count*0.75)
    valnum = math.floor(count*0.9)

    Base=os.path.join(output_folder_path,'data-split')
    os.mkdir(Base)
    os.mkdir(os.path.join(Base,'Scans_patch_mixed'))
    os.mkdir(os.path.join(Base,'Segmentation_patch_mixed'))
    os.mkdir(os.path.join(Base,'Scan_patch_test'))
    os.mkdir(os.path.join(Base,'Segmentation_patch_test'))
    os.mkdir('model_backup')
    os.mkdir('results')

    for i in range(0,trainnum):
        segName=segFiles[i]
        imageName=segName.split('Final_')[1]
        print('-------------------',imageName,'----------------------')
        group=DataFolders[0]
    
        getPatchMulticlass(group,imageName, segName, scans_path, seg_path,Base)
    for i in range(trainnum, valnum):
        segName=segFiles[i]
        imageName=segName.split('Final_')[1]
        print('-------------------',imageName,'----------------------')

        group=DataFolders[1]
        
        getPatchMulticlass(group,imageName, segName, scans_path, seg_path, Base)
    for i in range(valnum,count):
        segName=segFiles[i]
        imageName=segName.split('Final_')[1]
        print('-------------------',imageName,'----------------------')

        group=DataFolders[2]
        
        getPatchRand(group,imageName, segName, scans_path, seg_path, Base)
    print('Preprocessing Done')    
        
    #Training
    BS=200 
    dataPath=os.path.join(Base,'Scans_patch_mixed')
    segPath=os.path.join(Base,'Segmentation_patch_mixed')

    trainDataset=NPHDataset(dataPath,segPath,os.path.join(Base,'train_positions_mixed.txt'),Train=True)
    train_loader = DataLoader(trainDataset, batch_size=BS, num_workers=8, prefetch_factor=10000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)

    valDataset=NPHDataset(dataPath,segPath,os.path.join(Base,'val_positions_mixed.txt'),Train=False)
    val_loader = DataLoader(valDataset, batch_size=BS, num_workers=8, prefetch_factor=5000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)

    testDataPath=os.path.join(Base, 'Scan_patch_test')
    testSegPath=os.path.join(Base, 'Segmentation_patch_test')

    testDataset=NPHDataset(testDataPath,testSegPath,os.path.join(Base,'test_positions_rand.txt'),Train=False)
    test_loader = DataLoader(testDataset, batch_size=BS, num_workers=8, prefetch_factor=5000, pin_memory=True, drop_last=False, shuffle=True, persistent_workers=True)


    # 
    ResNet=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model = MyModel(ResNet,num_classes=4, num_outputs=4).to(device)
    optimizer =optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
    modelname='3Class_wd6_2Layer2x2_300'
    numEpoch=10

    print('start Training')
    for i in range(numEpoch):

        trainLoss, trainCorrect, trainTotal, TP, FN, FP=train(model,train_loader, optimizer, i, BS)
        with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
            file.write('Epoch {}, train loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, trainLoss, trainTotal, trainCorrect, trainCorrect/trainTotal))
        with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
            for j in range(1,4):
                file.write('Epoch {}, train dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))
        
        valLoss, valCorrect, valTotal, TP, FN, FP=test(model,i, val_loader, 'Val',BS)
        with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
            file.write('Epoch {}, val loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, valLoss, valTotal, valCorrect, valCorrect/valTotal))
        
        with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
            for j in range(1,4):
                file.write('Epoch {}, val dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))

        testLoss, testCorrect, testTotal, TP, FN, FP=test(model,i, test_loader, 'Test',BS)
        with open('results/result_data_{}.txt'.format(modelname),"a+") as file:
            file.write('Epoch {}, test loss: {}, total: {},correct: {}, accuracy: {:.6f}\n'.format(i, testLoss, testTotal, testCorrect, testCorrect/testTotal))

        with open('results/result_classwise_{}.txt'.format(modelname),"a+") as file:
            for j in range(1,4):
                file.write('Epoch {}, test dice score for class{}: {}\n'.format(i, j, 2*TP[j]/(2*TP[j]+FP[j]+FN[j])))

                            
        if (i)%5==0: 
            torch.save(model.state_dict(), "model_backup/epoch{}_2Dresnet{}.pt".format(i, modelname))  
        
    torch.save(model.state_dict(), "model_backup/epoch{}_ResNet2D{}.pt".format(i, modelname)) 
    outputs_path_dict = {}
    outputs_path_dict['Trained models'] = os.path.join(output_folder_path,'model_backup')
    outputs_path_dict['Dice Scores'] = os.path.join(output_folder_path,'results/result_classwise_{}.txt'.format(modelname))
    outputs_path_dict['Accuracy'] = os.path.join(output_folder_path,'results/result_data_{}.txt'.format(modelname))
    outputs_path_dict['Output Image'] = os.path.join(output_folder_path,'outputimg.jpg')
    return outputs_path_dict


if __name__ == '__main__':
    input_path_dict = {}
    outputs_path_dict = {}
    cwd = os.getcwd()
    print(cwd)
    input_path_dict['Raw CT Images'] = os.path.join(cwd,'Scans') 
    input_path_dict['Ground Truth Images'] = os.path.join(cwd,'Segmentation') 
    output_folder_path = cwd
    
    outputs_path_dict = run_module(input_path_dict, output_folder_path)
    print(outputs_path_dict)
