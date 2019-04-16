import preprocess
import sys
import pickle


def main(f_name):
    input_file = f_name
    # input_file = 'input-try.txt'
    mat = preprocess.main(input_file)
    with open('Classifier.pickle', 'rb') as f:
        clf = pickle.load(f)
    scores = clf.predict_proba(mat)
    dict_ = {}
    with open(input_file, 'r') as f:
        for row, val in zip(f, scores):
            seq = row.split()[1]
            dict_[seq] = val[-1]

    return (dict_)


if __name__ == "__main__":
    main(sys.argv[-1])
    # input_file=sys.argv[-1]
    # input_file='input-try.txt'
    # mat=preprocess.main(input_file)
    # with open('Classifier.pickle','rb') as f:
    #     clf=pickle.load(f)
    # scores=clf.predict_proba(mat)
    # dict_={}
    # with open(input_file,'r') as f:
    #     for row,val in zip(f,scores):
    #         seq=row.split()[1]
    #         dict_[seq]=val[-1]
    #
    # print(dict_)
