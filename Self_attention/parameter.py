import argparse
import parser
from texttable import Texttable

def parse_args():
    parser = argparse.ArgumentParser()  # 参数解析器对象

    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment')
    parser.add_argument('--exp_name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='DD', help='Dataset to use: [DD, PROTEINS, NCI1, etc]')
    parser.add_argument('--model', type=str, default='SAGPooling_Hierarchical', choices=['SAGPooling_Global', 'SAGPooling_Hierarchical'], help='Model to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of the training and validation batch')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Size of the testing batch')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--hid', type=int, default=128, help='Size of the hidden layer')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='Graph pooling ratio')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay of Adam')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')

    args=parser.parse_args()

    return args

class IOStream():
    def __init__(self,path):
        self.file=open(path,"a")

    def cprint(self,text):
        print(text)
        self.file.write(text+"\n")
        self.file.flush()

    def close(self):
        self.file.close()

def table_printer(args):
    args=vars(args)
    keys=sorted(args.keys())
    table=Texttable()
    table.set_cols_dtype(["t","t"])
    rows=[["Parameter","Value"]]
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])])
    table.add_rows(rows)
    return table.draw()
