"""Reformats COGS dataset as input formats required by OpenNMT."""

import argparse
import os
from collections import Counter
import random
# Prespecified set of file names in the COGS dataset.
_DATASET_FILENAMES = ['train', 'test', 'dev', 'gen', 'train_100']

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default=None, type=str, required=True,
                        help='Path to directory containing the COGS dataset.')
    parser.add_argument('--output_path', default=None, type=str, required=True,
                        help='Path to save the output data to.')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed for selecting a small subset of test data as validation set')
    args = parser.parse_args()

    source_vocab = Counter()
    target_vocab = Counter()

    # Create a new directory if the specified output path does not exist.
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for filename in _DATASET_FILENAMES:
        with open(os.path.join(args.input_path, f'{filename}.tsv')) as f:
            data = f.readlines()

        source_lines = []
        target_lines = []
        code_lines = []
        for line in data:
            source, target, code = line.rstrip('\n').split('\t')
            source_lines.append('bos {}\n'.format(source))
            target_lines.append('bos {}\n'.format(target))
            code_lines.append('{}\n'.format(code))
            source_vocab.update(source.split())
            target_vocab.update(target.split())

        # Write the datapoints to source and target files.
        with open(os.path.join(args.output_path, f'{filename}.word'), 'w') as wf:
            wf.writelines(source_lines)

        with open(os.path.join(args.output_path, f'{filename}.predicate'), 'w') as wf:
            wf.writelines(target_lines)

        with open(os.path.join(args.output_path, f'{filename}.code'), 'w') as wf:
            wf.writelines(code_lines)


        #sample a small validation set
        if filename == 'gen':
            items = list(zip(source_lines, target_lines, code_lines))
            random.seed(args.seed)
            random.shuffle(items)
            items = items[:len(items)//10]
            filename = "gen-dev"
            source_lines = [item[0] for item in items]
            target_lines = [item[1] for item in items]
            code_lines = [item[2] for item in items]
            with open(os.path.join(args.output_path, f'{filename}.word'), 'w') as wf:
                wf.writelines(source_lines)

            with open(os.path.join(args.output_path, f'{filename}.predicate'), 'w') as wf:
                wf.writelines(target_lines)

            with open(os.path.join(args.output_path, f'{filename}.code'), 'w') as wf:
                wf.writelines(code_lines)


    # Write the vocabulary files.
    with open(os.path.join(args.output_path, 'source_vocab.txt'), 'w') as wf:
        for w in list(source_vocab.most_common()):
            wf.write(w[0]+" "+str(w[1]))
            wf.write('\n')

    with open(os.path.join(args.output_path, 'target_vocab.txt'), 'w') as wf:
        for w in list(target_vocab.most_common()):
            wf.write(w[0]+" "+str(w[1]))
            wf.write('\n')

    print(f'Reformatted and saved COGS data to {args.output_path}.')


if __name__ == '__main__':
    main()
