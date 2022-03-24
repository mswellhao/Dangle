from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.data import encoders
from fairseq.logging.meters import StopwatchMeter, TimeMeter

import torch
import sys, os, math, glob, re, random
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from collections import Counter

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='semantic_parsing')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)

print(args)
print(args.max_sentences)
print(args.max_tokens)
# Set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary
src_dict = task.source_dictionary


if args.results_path is not None:
    os.makedirs(args.results_path, exist_ok=True)
    output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
    output_file = open(output_path, 'w', buffering=1, encoding='utf-8')
else:
    output_file = sys.stdout

code_file_path = os.path.join(args.data, "{}.word-predicate.code".format(args.gen_subset))
if os.path.exists(code_file_path):
    with open(code_file_path) as f:
        codes = [line.strip() for line in f.readlines()]
else:
    codes = None


use_cuda = torch.cuda.is_available() and not args.cpu
# Load ensemble
print('loading model(s) from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble(
    utils.split_paths(args.path),
    arg_overrides=eval(args.model_overrides),
    task=task,
    suffix=getattr(args, "checkpoint_suffix", ""),
)
# Optimize ensemble for generation
for model in models:
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

# Load dataset (possibly sharded)
itr = task.get_batch_iterator(
    dataset=task.dataset(args.gen_subset),
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    ),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    num_shards=args.num_shards,
    shard_id=args.shard_id,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)
progress = progress_bar.progress_bar(
    itr,
    log_format=args.log_format,
    log_interval=args.log_interval,
    default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
)


# Initialize generator
gen_timer = StopwatchMeter()
generator = task.build_generator(models, args)

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x



type_num, type_correct_num = Counter(), Counter()
correct, wrong, isum = 0, 0, 0
correct_examples, wrong_examples = [], []
gid = 0
for eid, sample in enumerate(progress):
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    if 'net_input' not in sample:
        continue

    prefix_tokens = None
    if args.prefix_size > 0:
        prefix_tokens = sample['target'][:, :args.prefix_size]
    
    hypos = task.inference_step(generator, models, sample, prefix_tokens)

    gen_timer.start()
    hypos = task.inference_step(generator, models, sample, prefix_tokens)
    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
    gen_timer.stop(num_generated_tokens)

    if eid % 100 == 0:
        print("Finish %d examples " % eid)

    for i, sample_id in enumerate(sample['id'].tolist()):
        has_target = sample['target'] is not None

        # Remove padding
        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
        target_tokens = None
        if has_target:
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

        # Either retrieve the original sentences or regenerate them from tokens.
        if src_dict is not None:
            src_str = src_dict.string(src_tokens)
        else:
            src_str = ""

        if has_target:
            target_str = tgt_dict.string(
                target_tokens,
                escape_unk=True,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
        if args.source_bpe_decode:
            detok_src_str = decode_fn(src_str)
        else:
            detok_src_str = src_str

        if args.target_bpe_decode and has_target:
            detok_target_str = decode_fn(target_str)
        else:
            detok_target_str = target_str
    
    
        if not args.quiet:
            print('\n', file=output_file)
            print(str(gid), file=output_file)
           
            if codes is not None:
                print('code : {}'.format(codes[sample_id]), file=output_file)

            if src_dict is not None:
                print('S-token\t{}'.format(src_tokens), file=output_file)
                print('S\t{}'.format(src_str), file=output_file)
                print('S-decode\t{}\t{}'.format(sample_id, detok_src_str), file=output_file)
            if has_target:
                print('T-token\t{}'.format(target_tokens), file=output_file)
                print('T\t{}'.format(target_str), file=output_file)
                print('T-decode\t{}\t{}'.format(sample_id, detok_target_str), file=output_file)



        # Process top predictions
        for j, hypo in enumerate(hypos[i][:args.nbest]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'],
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                extra_symbols_to_ignore={
                    generator.eos,
                }
            )
            hypo_str = tgt_dict.string(
                    hypo_tokens,
                    escape_unk=True,
                    extra_symbols_to_ignore={
                        generator.eos,
                    }
                )
            if args.target_bpe_decode:
                detok_hypo_str = decode_fn(hypo_str)
            else:
                detok_hypo_str = hypo_str

            if codes is not None:
                if detok_target_str == detok_hypo_str:
                    type_correct_num[codes[sample_id]] += 1
                type_num[codes[sample_id]] += 1
                
            if not args.quiet:
                if detok_target_str == detok_hypo_str:
                    print('Correct', file=output_file)
                else:
                    print('Incorrect', file=output_file)

                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                # detokenized hypothesis
                print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                print('P-{}\t{}'.format(
                    sample_id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ), file=output_file)

                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        sample_id,
                        ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                    ), file=output_file)

                if args.print_step:
                    print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                if getattr(args, 'retain_iter_history', False):
                    for step, h in enumerate(hypo['history']):
                        _, h_str, _ = utils.post_process_prediction(
                            hypo_tokens=h['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=None,
                            align_dict=None,
                            tgt_dict=tgt_dict,
                            remove_bpe=None,
                        )
                        print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

            # Score only the top hypothesis
            if has_target and j == 0:
                if detok_target_str == detok_hypo_str:
                    correct += 1
                    # correct_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                    # print("correct_examples")
                else:
                    wrong += 1
                    # wrong_examples.append((detok_target_str, detok_hypo_str, detok_src_str))
                isum += 1

        gid += 1


print("Model : "+args.path+" ;  Accuracy : "+str(correct/isum), file=output_file)
if codes is not None:
    print("Code type Accuracy : "+"  ;  ".join([t+" : "+str(type_correct_num[t]/type_num[t]) for t in type_num]), file=output_file)

                    

