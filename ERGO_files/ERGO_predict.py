import torch
import argparse
import ERGO_files.ae_utils as ae
import ERGO_files.lstm_utils as lstm
from ERGO_files.ERGO_models import AutoencoderLSTMClassifier, DoubleLSTMClassifier
import csv
import os

def predict(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    path = os.path.abspath('ERGO_files')
    if args.ae_file == 'auto':
        args.ae_file = path + '/tcr_ae_dim_100.pt'
    if args.model_file == 'auto':
        args.model_file = path + '/' + '_'.join([args.model_type, args.dataset]) + '1.pt'
    # Read test data
    tcrs = []
    peps = []
    signs = []
    max_len = 28
    with open(args.test_data_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            tcr, pep = line
            if args.model_type == 'ae' and len(tcr) >= max_len:
                continue
            tcrs.append(tcr)
            peps.append(pep)
            signs.append(0.0)
    tcrs_copy = tcrs.copy()
    peps_copy = peps.copy()
    # Load model
    device = args.device
    if args.model_type == 'ae':
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 100, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        model = DoubleLSTMClassifier(10, 500, 0.1, device)
        checkpoint = torch.load(args.model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    # Predict
    batch_size = 50
    if args.model_type == 'ae':
        test_batches = ae.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_len)
        preds = ae.predict(model, test_batches, device)
    if args.model_type == 'lstm':
        lstm.convert_data(tcrs, peps, amino_to_ix)
        test_batches = lstm.get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix)
        preds = lstm.predict(model, test_batches, device)
    # Print predictions
    # for tcr, pep, pred in zip(tcrs_copy, peps_copy, preds):
    #     print('\t'.join([tcr, pep, str(pred)]))
    return tcrs_copy, peps_copy, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("dataset")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--ae_file", default='auto')
    parser.add_argument("--model_file", default='auto')
    parser.add_argument("--test_data_file")
    args = parser.parse_args()
    predict(args)

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
