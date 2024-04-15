import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import logging
log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
             "%(filename)s::%(lineno)d::%(message)s"
logging.basicConfig(level='DEBUG', format=log_format)

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.utils.data import random_split
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




class HistDataset(Dataset):
    def __init__(self, file_paths):
        # file_paths = np.array(sorted(os.listdir('/scratch/shubham.ojha/WSI/tcga_brain/feat_dir')))
        self.file_paths = file_paths
                
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        x = data[0]
        y = data[1]  
        return x, y
    


def train(train_loader, milnet, criterion, optimizer, args):
    milnet.train()
    total_loss = 0
    i = 0
    Tensor = torch.cuda.FloatTensor
    for data in train_loader:
        logging.info(f'train_index :- {i}')
        optimizer.zero_grad()
        feats, label = data
        feats = feats.to('cuda', dtype=torch.float32)
        label = label.to('cuda', dtype=torch.float32)
        bag_feats = feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)
        #logging.info(f'bag_prediction.view(1, -1) :- {bag_prediction.view(1, -1)}')
        #logging.info(f'label.view(1, -1) :- {label.view(1, -1)}')
        #logging.info(f'max_prediction.view(1, -1) :- {max_prediction.view(1, -1)}')
        bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        i += 1
    return total_loss / len(train_loader)


def val(val_loader, milnet, criterion, optimizer, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    i = 0
    with torch.no_grad():
        for data in val_loader:
            logging.info(f'val_index :- {i}')
            i += 1
            feats, label = data
            feats, label = data
            feats = feats.to('cuda', dtype=torch.float32)
            label = label.to('cuda', dtype=torch.float32)
            bag_feats = feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    #test_labels = np.array(test_labels)
    #test_labels = np.array([label.cpu().numpy() for label in test_labels])
    #logging.info(f'test_predictions_before :- {test_predictions}')
    test_predictions = np.array(test_predictions)
    #logging.info(f'test_predictions_before :- {test_predictions}')

    #test_labels = np.array(test_labels)
    
    #logging.info(f'test_labels_before :- {test_labels}')
    
    test_labels = [test_label.cpu() for test_label in test_labels]
    test_labels = [test_label.numpy()[0] for test_label in test_labels]
    test_labels = np.array(test_labels)
    
    #logging.info(f'test_labels_after :- {test_labels}')

    
    
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    #logging.info(f'auc_value :- {auc_value}')
    #logging.info(f'sum_auc_value :- {sum(auc_value)}')
    

    
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    #logging.info(f'test_predictions :- {test_predictions}')


    for i in range(0, len(val_loader)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(val_loader)
    #logging.info(f'avg_score :- {avg_score}')

    total_F1 = 0
    for i in range(0, len(val_loader)):
        total_F1 = f1_score(test_labels[i], test_predictions[i]) + total_F1

    avg_F1 = total_F1/ len(val_loader)
    #logging.info(f'avg_F1 :- {avg_F1}')

    avg_AUC = sum(auc_value)/args.num_classes
    
    return total_loss / len(val_loader), avg_score, avg_AUC, thresholds_optimal, avg_F1






def test(test_loader, milnet, criterion, optimizer, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    i = 0
    with torch.no_grad():
        for data in test_loader:
            logging.info(f'test_index :- {i}')
            i += 1
            feats, label = data
            feats, label = data
            feats = feats.to('cuda', dtype=torch.float32)
            label = label.to('cuda', dtype=torch.float32)
            bag_feats = feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            
            #bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss = nn.BCEWithLogitsLoss()
            bag_loss = loss(bag_prediction.view(1, -1), label.view(1, -1))
            max_loss = loss(max_prediction.view(1, -1), label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = [test_label.cpu() for test_label in test_labels]
    test_labels = [test_label.numpy()[0] for test_label in test_labels]
    test_labels = np.array(test_labels)
    #test_labels = np.array([label.cpu().numpy() for label in test_labels])
    
    test_predictions = np.array(test_predictions)
    
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)

    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0

    for i in range(0, len(test_loader)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_loader)
    #logging.info(f'avg_score :- {avg_score}')

    total_F1 = 0
    for i in range(0, len(test_loader)):
        total_F1 = f1_score(test_labels[i], test_predictions[i]) + total_F1

    avg_F1 = total_F1/ len(test_loader)
    #logging.info(f'avg_F1 :- {avg_F1}')

    avg_AUC = sum(auc_value)/args.num_classes
    
    return total_loss / len(test_loader), avg_score, avg_AUC, thresholds_optimal, avg_F1

    

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        #logging.info(f'c :- {c}')

        label = labels[:, c]
        #logging.info(f'label :- {label}')

        prediction = predictions[:, c]
        #logging.info(f'prediction :- {prediction}')

        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        #fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        logging.info(f'label :- {label}')
        logging.info(f'prediction :- {prediction}')
        c_auc = roc_auc_score(label, prediction)
        

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    #parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    #parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('-p', '--path_to_splits', type=str, default='/scratch/shubham.ojha/WSI', help = 'Path to Splits folder')
    parser.add_argument('--starting_range_fold', type=int, default=0, help = 'starting_range_fold')
    parser.add_argument('--end_range_fold', type=int, default=10, help = 'end_range_fold')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=2).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=2, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    if args.model == 'dsmil':
        state_dict_weights = torch.load('init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            milnet.load_state_dict(state_dict_weights, strict=False)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    

    val_metrices = {
        'fold' : [],
        'val_loss_bag' : [],
        'val_avg_score' : [],
        'val_aucs' : [],
        'val_avg_F1': []
        }
    test_metrices = {
        'fold' : [],
        'test_loss_bag' : [],
        'test_avg_score' : [],
        'test_aucs' : [],
        'test_avg_F1': []
    } 
    for fold in range(args.starting_range_fold, args.end_range_fold):
        logging.info(f'fold :-{fold} ')
        
        path = args.path_to_splits
        test_path = path + 'test' + str(fold) + '.csv'
        train_path = path + 'train' + str(fold) + '.csv'
        val_path = path + 'val' + str(fold) + '.csv'
        
        
        




        train_dataset = HistDataset(pd.read_csv(train_path)['path'].tolist())
        val_dataset = HistDataset(pd.read_csv(val_path)['path'].tolist())
        test_dataset = HistDataset(pd.read_csv(test_path)['path'].tolist())


    
        train_loader = DataLoader(train_dataset, batch_size = 1)
        val_loader = DataLoader(val_dataset, batch_size = 1)
        test_loader = DataLoader(test_dataset, batch_size = 1)


        
        
        
        patience = 20
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(1, args.num_epochs):
            logging.info(f'epoch :-{epoch} ')
            train_loss_bag = train(train_loader, milnet, criterion, optimizer, args)
            # val(val_loader, milnet, criterion, optimizer, args)
            logging.info(f'Validation')
            val_loss_bag, val_avg_score, val_aucs, _, val_avg_F1 = val(val_loader, milnet, criterion, optimizer, args)
            logging.info(f'Val metrices')
            logging.info(f'epoch - {epoch}, loss - {val_loss_bag}, accuracy:- {val_avg_score}, avg_AUC - {val_aucs}, F1 - {val_avg_F1}')

            scheduler.step()

            # Check for early stopping
            
            if epoch>=35:
                logging.info(f'Epoch above 35')
                logging.info(f'epoch :- {epoch}')
                if val_loss_bag < best_val_loss:
                    best_val_loss = val_loss_bag
                    epochs_without_improvement = 0
                    # Save the state dictionary of the best model
                    best_model_state = milnet.state_dict()
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch+1} epochs without improvement.")
                    break
            # Load the best model state dictionary for testing
            if best_model_state is not None:
                milnet.load_state_dict(best_model_state)
            
        
        
        logging.info(f'Testing')
        test_loss_bag, test_avg_score, test_aucs, _, test_avg_F1 = test(test_loader, milnet, optimizer, criterion, args)
        logging.info(f'Test metrices')
        logging.info(f'epoch - {epoch}, loss - {test_loss_bag}, accuracy:- {test_avg_score}, avg_AUC - {test_aucs}, F1 - {test_avg_F1}')

        val_metrices['fold'].append(fold)
        val_metrices['val_loss_bag'].append(val_loss_bag)
        val_metrices['val_avg_score'].append(val_avg_score)
        val_metrices['val_aucs'].append(val_aucs)
        val_metrices['val_avg_F1'].append(val_avg_F1)

        test_metrices['fold'].append(fold)
        test_metrices['test_loss_bag'].append(test_loss_bag)
        test_metrices['test_avg_score'].append(test_avg_score)
        test_metrices['test_aucs'].append(test_aucs)
        test_metrices['test_avg_F1'].append(test_avg_F1)

        


    
    
    val_df = pd.DataFrame(val_metrices)
    val_directory_path = '/scratch/shubham.ojha/WSI/tcga_brain/' + str(args.starting_range_fold)
    os.makedirs(val_directory_path, exist_ok=True)
    csv_val_df_path = '/scratch/shubham.ojha/WSI/tcga_brain/' + str(args.starting_range_fold) + '/val_metrices.csv'
    val_df.to_csv(csv_val_df_path, index=False) 

    test_df = pd.DataFrame(test_metrices)
    csv_test_df_path = '/scratch/shubham.ojha/WSI/tcga_brain/' + str(args.starting_range_fold) + '/test_metrices.csv'
    test_df.to_csv(csv_test_df_path, index=False)
    


            

if __name__ == '__main__':
    main()