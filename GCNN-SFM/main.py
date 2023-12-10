# This is an example to train a two-classes model.

import torch
import Biodata, CNmodel
device = torch.device('cuda')
# 进程
from multiprocessing import Process


def foo(i):
        print(" This is Process ", i)


def main():
        for i in range(5):
                p = Process(target=foo, args=(i,))
                p.start()


if __name__ == '__main__':
        #############    Train   ########################
        main()
        # data = Biodata.Biodata(fasta_file=".\Data\\e\\train.fasta",
        #                        label_file=".\Data\\e\\train_label.txt",
        #                        feature_file= None
        #                        )
        data = Biodata.Biodata(fasta_file=".\\Data\\H\\train.fasta",
                               label_file=".\\Data\\H\\train_label.txt",
                               feature_file=None
                               )
        # data = Biodata.Biodata(fasta_file=".\\Data\\m\\train.fasta",
        #                        label_file=".\\Data\\m\\train_label.txt",
        #                        feature_file=None
        #                        )
        # # data = Biodata.Biodata(fasta_file=".\Data\DNN-EG\\train.fasta",
        # #                        label_file=".\Data\DNN-EG\\train_label.txt",
        # #                        feature_file=None
        # #                        )
        dataset = data.encode(thread=20)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNmodel.model(label_num=2, other_feature_dim=0).to(device)
        CNmodel.train(dataset, model, weighted_sampling=True)
        # ##############   Test    ####################
        # data2 = Biodata.Biodata(fasta_file=".\Data\\e\\test.fasta",
        #                        label_file=".\Data\\e\\test_label.txt",
        #                        feature_file= None
        #                        )
        # data2 = Biodata.Biodata(fasta_file=".\\Data\\H\\test.fasta",
        #                        label_file=".\\Data\\H\\test_label.txt",
        #                        feature_file=None
        #                        )
        # data2 = Biodata.Biodata(fasta_file=".\\Data\\m\\test.fasta",
        #                        label_file=".\\Data\\m\\test_label.txt",
        #                        feature_file=None
        #                        )
        # data2 = Biodata.Biodata(fasta_file=".\Data\DNN-EG\\train.fasta",
        #                        label_file=".\Data\DNN-EG\\train_label.txt",
        #                        feature_file=None
        #                        )
        # torch.cuda.empty_cache()
        # data1 = data2.encode(thread=20)
        # CNmodel.test(data1)






