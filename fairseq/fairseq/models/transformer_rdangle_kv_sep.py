# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerDecoderLayerKVDangle,
    TransformerEncoderLayer,
    TransformerEncoderLayerCrossBoundary,
    GradMultiply,
)
from typing import NamedTuple

from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.roberta.model import RobertaModel


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_out_value", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("src_attns", Optional[List[Tensor]]),  # List[B x T x T]

    ],
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_rdangle_kv_sep")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
     
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--dangle-encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')


        parser.add_argument('--use-rel-pos', default=False,
                            action='store_true',
                            help='use relative position attention')
        parser.add_argument('--max-relative-position', type=int, default=64,
                            help='number of positional embeddings to learn')


        
        parser.add_argument('--chunk-size', type=int, metavar='N', default=1,
                            help='chunk size')
        parser.add_argument('--key-enc-scale', type=float, metavar='D',default=1,
                            help='scale for disentangled encodings')
        parser.add_argument('--value-enc-scale', type=float, metavar='D',default=1,
                            help='scale for disentangled encodings')

        parser.add_argument('--dis-encoder-layers', type=int, metavar='N',
                            help='num dis encoder layers')
        parser.add_argument('--kv-encoder-layers', type=int, metavar='N', default=0,
                            help='num kv encoder layers')
        


        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )


        if args.decoder_layerdrop > 0.0:
            rep_shared_layers = LayerDropModuleList(p=args.decoder_layerdrop)
        else:
            rep_shared_layers = nn.ModuleList([])
        rep_shared_layers.extend(
            [
                TransformerEncoderLayer(args)
                for _ in range(args.encoder_layers)
            ]
        )

        if args.decoder_layerdrop > 0.0:
            value_dis_encoder_layers = LayerDropModuleList(p=args.decoder_layerdrop)
        else:
            value_dis_encoder_layers = nn.ModuleList([])
        value_dis_encoder_layers.extend(
            [
                TransformerEncoderLayerCrossBoundary(args)
                for _ in range(args.dis_encoder_layers)
            ]
        )

        
        print("Not sharing dis encoder")
        if args.decoder_layerdrop > 0.0:
            key_dis_encoder_layers = LayerDropModuleList(p=args.decoder_layerdrop)
        else:
            key_dis_encoder_layers = nn.ModuleList([])
        key_dis_encoder_layers.extend(
            [
                TransformerEncoderLayerCrossBoundary(args)
                for _ in range(args.dis_encoder_layers)
            ]
        )


        if args.decoder_layerdrop > 0.0:
            param_shared_layers = LayerDropModuleList(p=args.decoder_layerdrop)
        else:
            param_shared_layers = nn.ModuleList([])
        param_shared_layers.extend(
            [
                TransformerEncoderLayer(args)
                for _ in range(args.kv_encoder_layers)
            ]
        )

        encoder = cls.build_encoder(args, rep_shared_layers, value_dis_encoder_layers, param_shared_layers, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, rep_shared_layers, key_dis_encoder_layers, param_shared_layers, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, rep_shared_layers, value_dis_encoder_layers, param_shared_layers, src_dict, embed_tokens):
        return TransformerEncoder(
            args, 
            rep_shared_layers, 
            value_dis_encoder_layers, 
            param_shared_layers, 
            src_dict, 
            embed_tokens)

    @classmethod
    def build_decoder(cls, args, rep_shared_layers, key_dis_encoder_layers, param_shared_layers, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            rep_shared_layers, 
            key_dis_encoder_layers, 
            param_shared_layers,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )


        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, rep_shared_layers, value_dis_encoder_layers, param_shared_layers, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        # d = self.embed_tokens.weight.data
        # print("average embed_tokens norm : "+str((d**2).sum() / d.size(0)))

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.src_emb = nn.Embedding(1, embed_dim)
        nn.init.normal_(self.src_emb.weight, mean=0, std=0.02)

        self.rep_shared_layers = rep_shared_layers
        self.value_dis_encoder_layers = value_dis_encoder_layers
        self.param_shared_layers = param_shared_layers

        if args.encoder_normalize_before:
            self.value_layer_norm = LayerNorm(embed_dim)
        else:
            self.value_layer_norm = None

        #shared encoding token
        self.se_emb = nn.Embedding(1, embed_dim)
        nn.init.normal_(self.se_emb.weight, mean=0, std=0.02)

        # d = self.src_emb.weight.data
        # print("average src_emb norm : "+str((d**2).sum() / d.size(0)))

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        type_embed = self.src_emb(torch.zeros_like(src_tokens))
        x += type_embed
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        source_x_value = x
        for idx, layer in enumerate(self.rep_shared_layers):
            source_x_value = layer(source_x_value, encoder_padding_mask)

        #add special token to be compatable input to dis-encoder layer
        se_token_embed = self.se_emb(torch.zeros_like(src_tokens[:,-1:])).transpose(0,1)
        # (T+1) x B x C 
        source_x_value = torch.concat([source_x_value, se_token_embed], dim=0)
        # B x (T+1)
        se_encoder_padding_mask = torch.cat([encoder_padding_mask, torch.zeros_like(encoder_padding_mask[:,:1])], dim=1)
        
        _, src_T = src_tokens.shape
        for i, layer in enumerate(self.value_dis_encoder_layers):
            source_x_value = layer(source_x_value, se_encoder_padding_mask, part1_len = src_T, part2_len = 1)

        #T x B x C
        source_x_value = source_x_value[:src_T]
        for idx, layer in enumerate(self.param_shared_layers):
            source_x_value = layer(source_x_value, encoder_padding_mask)

        if self.value_layer_norm is not None:
            source_x_value = self.value_layer_norm(source_x_value)


        
        # print(f"value layer norm : {self.value_layer_norm.weight.norm()}")



        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_out_value=source_x_value,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None, # B x T
            src_lengths=None, # B x 1
            src_attns=None, # List[B x T x T]
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out_value = (
            encoder_out.encoder_out_value
            if encoder_out.encoder_out_value is None
            else encoder_out.encoder_out_value.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        src_attns = encoder_out.src_attns
        if src_attns is not None:
            for idx, state in enumerate(src_attns):
                src_attns[idx] = state.index_select(0, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_out_value=new_encoder_out_value, # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
            src_attns=src_attns,  # List[B x T x T]
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, rep_shared_layers, key_dis_encoder_layers, param_shared_layers, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos
            )
            if not args.no_token_positional_embeddings
            else None
        )


        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.rep_shared_layers = rep_shared_layers
        self.key_dis_encoder_layers = key_dis_encoder_layers
        self.param_shared_layers = param_shared_layers


        if args.encoder_normalize_before and len(self.rep_shared_layers) > 0:
            self.shared_norm = LayerNorm(embed_dim)
        else:
            self.shared_norm = None

        if args.encoder_normalize_before:
            self.key_layer_norm = LayerNorm(embed_dim)
        else:
            self.key_layer_norm = None


        if self.decoder_layerdrop > 0.0:
            self.decoder_layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.extend(
            [
                TransformerDecoderLayerKVDangle(args)
                for _ in range(args.decoder_layers)
            ]
        )

        self.num_layers = len(self.param_shared_layers)


        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )


        self.tgt_emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.tgt_emb.weight, mean=0, std=0.02)

        self.output_hiddens = getattr(args, "output_hiddens", False)
        self.chunk_size = getattr(args, "chunk_size", 1)


        self.split_num = getattr(args, "split_num", 1)

        self.key_enc_scale = getattr(args, "key_enc_scale", 1)
        self.value_enc_scale = getattr(args, "value_enc_scale", 1)





    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, out_hidden_states = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)

        if out_hidden_states is not None:
            out = {"hiddens":out_hidden_states}
            return (x, out)
        else:
            return (x,)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

   

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens
            )
            if self.embed_positions is not None
            else None
        )

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        type_embed = self.tgt_emb(torch.zeros_like(prev_output_tokens))
        x += type_embed

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)
        # B x T 
        target_padding_mask = prev_output_tokens.eq(self.padding_idx)
        # T x B x C
        source_x = encoder_out.encoder_out
        src_T, _, _ = source_x.shape
        # B x T 
        source_padding_mask = encoder_out.encoder_padding_mask
        #compute shared reps
        #T x B x C
        for idx, layer in enumerate(self.rep_shared_layers):
            source_x = layer(source_x, source_padding_mask)

        if self.shared_norm is not None:
            source_x = self.shared_norm(source_x)

        source_x_value = encoder_out.encoder_out_value
        source_x_value = self.value_enc_scale * source_x_value

        # def print_grad_norm_key(grad):
        #     print(f"key grad norm : {grad.norm()}")

        # def print_grad_norm_value(grad):
        #     print(f"value grad norm : {grad.norm()}")

        B, T, C = x.shape
        if incremental_state is None:
            out_x = []
            token_sum_total = sum([i for i in range(1,T+1,self.chunk_size)])
            batch_token_num = token_sum_total // self.split_num + 1
            subbatches = [[]]
            for i in range(1,T+1,self.chunk_size):
                subbatches[-1].append(i)
                if sum(subbatches[-1]) >= batch_token_num:
                    subbatches.append([])
            # print(f"T : {T} ; subbatches : {subbatches}")
            if len(subbatches[-1]) == 0:
                subbatches = subbatches[:-1]
            for subbatch in subbatches:
                #1. compute adaptive source reps

                #T x m*B x C
                stack_source_x = torch.concat([source_x]*len(subbatch),dim=1)
                #m*B x T
                stack_source_padding_mask = torch.concat([source_padding_mask]*len(subbatch),dim=0)
                #T x m*B x C
                stack_source_x_value = torch.concat([source_x_value]*len(subbatch),dim=1)

                #m*B x T x C
                stack_dis_x = x.new(B*len(subbatch), subbatch[-1], C).fill_(0)
                #m*B x T
                stack_dis_padding_mask = target_padding_mask.new(B*len(subbatch), subbatch[-1]).fill_(1)
                for i, idx in enumerate(subbatch):
                    stack_dis_x[i*B:(i+1)*B, -idx:] = x[:,:idx]
                    stack_dis_padding_mask[i*B:(i+1)*B, -idx:] = target_padding_mask[:,:idx]

                end_token_embed = self.tgt_emb(torch.ones_like(stack_dis_padding_mask[:,-1:], dtype=torch.long))
                stack_dis_x = torch.cat([stack_dis_x, end_token_embed], dim=1)
                stack_dis_padding_mask = torch.cat([stack_dis_padding_mask, torch.zeros_like(stack_dis_padding_mask[:,:1])], dim=1)                   
                stack_dis_x = stack_dis_x.transpose(0, 1)

                stack_dis_x = torch.cat([stack_source_x, stack_dis_x], dim=0)
                stack_dis_padding_mask = torch.cat([stack_source_padding_mask, stack_dis_padding_mask], dim=1)

                for idx, layer in enumerate(self.key_dis_encoder_layers):
                    stack_dis_x = layer(stack_dis_x, stack_dis_padding_mask, part1_len = src_T, part2_len = stack_dis_x.size(0)-src_T)
                
                # T x B x C
                stack_dis_source_x_key = stack_dis_x[:src_T]
                for idx, layer in enumerate(self.param_shared_layers):
                    stack_dis_source_x_key = layer(stack_dis_source_x_key, stack_source_padding_mask)

                if self.key_layer_norm is not None:
                    stack_dis_source_x_key = self.key_layer_norm(stack_dis_source_x_key)

                stack_dis_source_x_key = self.key_enc_scale * stack_dis_source_x_key

                #2. decoding
                max_len = min(subbatch[-1]+self.chunk_size-1, T)
                #m*B x T x C
                stack_target_x = x.new(B*len(subbatch), max_len, C).fill_(0)
                 #m*B x T
                stack_target_padding_mask = target_padding_mask.new(B*len(subbatch), max_len).fill_(1)
                for i, idx in enumerate(subbatch):
                    ex_len = min(idx+self.chunk_size-1, T)
                    stack_target_x[i*B:(i+1)*B, -ex_len:] = x[:,:ex_len]
                    stack_target_padding_mask[i*B:(i+1)*B, -ex_len:] = target_padding_mask[:,:ex_len]

                
                # (i + chunk_size) x B x C
                stack_target_x = stack_target_x.transpose(0, 1)
                self_attn_mask = self.buffered_future_mask(stack_target_x)

                # print(f"key layer norm : {self.key_layer_norm.weight.norm()}")
                # print(f"value norm : {stack_source_x_value.norm()}")
                # print(f"key norm : {stack_dis_source_x_key.norm()}")
                # stack_source_x_value.register_hook(print_grad_norm_value)
                # stack_dis_source_x_key.register_hook(print_grad_norm_key)

                #T x m*B x C
                for idx, layer in enumerate(self.decoder_layers):
                    stack_target_x, _, _ = layer(
                        stack_target_x,
                        encoder_out_value = stack_source_x_value,
                        encoder_out_key = stack_dis_source_x_key,
                        encoder_padding_mask = stack_source_padding_mask,
                        incremental_state = None,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=stack_target_padding_mask,
                        need_attn=False,
                        need_head_weights=False,
                    )

                for i, idx in enumerate(subbatch):
                    if idx+self.chunk_size-1 > T:
                        decode_chunk_size = T - idx + 1
                    else:
                        decode_chunk_size = self.chunk_size

                    #chunk x B x C
                    chunk_output = stack_target_x[-decode_chunk_size:,i*B:(i+1)*B]
                    out_x.append(chunk_output)
            
            # T x B x C  
            out_x = torch.concat(out_x, dim=0)
        else:
            if positions is not None:
                target_x = x + positions
                dis_x = x + torch.flip(positions, dims=[1])
            else:
                target_x = x
                dis_x = x 

            # if T > 30:
            #     self.chunk_size = 32
            # else:
            #     self.chunk_size = 2
            ##1. compute adaptive source reps
            # B x T+1 x C
            if (T - 1) % self.chunk_size == 0: 
                dis_padding_mask = target_padding_mask
                #B x 1 x C
                end_token_embed = self.tgt_emb(torch.ones_like(prev_output_tokens[:,-1:]))        
                dis_x = torch.cat([dis_x, end_token_embed], dim=1)
                # B x T+1 x C -> T+1 x B x C
                dis_x = dis_x.transpose(0, 1)
                dis_padding_mask = torch.cat([dis_padding_mask, torch.zeros_like(dis_padding_mask[:,:1])], dim=1)
                    
                # (T+T+1) x B x C
                dis_x = torch.cat([source_x, dis_x], dim=0)
                # B x (T+T+1)
                dis_padding_mask = torch.cat([source_padding_mask, dis_padding_mask], dim=1)
                for idx, layer in enumerate(self.key_dis_encoder_layers):
                    # (T+T+1) x B x C
                    # print(f"dis layer {idx} : ")
                    dis_x = layer(dis_x, dis_padding_mask, part1_len = src_T, part2_len = dis_x.size(0)-src_T)

                # T x B x C
                dis_source_x_key = dis_x[:src_T]
                for idx, layer in enumerate(self.param_shared_layers):
                    dis_source_x_key = layer(dis_source_x_key, source_padding_mask)

                if self.key_layer_norm is not None:
                    dis_source_x_key = self.key_layer_norm(dis_source_x_key)

                dis_source_x_key = self.key_enc_scale * dis_source_x_key

                for key in list(incremental_state.keys()):
                    del incremental_state[key]

                # B x T x C -> T x B x C
                target_x = target_x.transpose(0, 1)
                self_attn_mask = self.buffered_future_mask(target_x)

            else:
                target_x = target_x[:,-1:]
                target_padding_mask = target_padding_mask[:,-1:]
                # B x 1 x C -> 1 x B x C
                target_x = target_x.transpose(0, 1)
                self_attn_mask = None
                source_x_value = None
                dis_source_x_key = None

            ##2. decoding
            for idx, layer in enumerate(self.decoder_layers):
                #T x B x C
                target_x, _, _ = layer(
                    target_x,
                    encoder_out_value = source_x_value,
                    encoder_out_key = dis_source_x_key,
                    encoder_padding_mask = source_padding_mask,
                    incremental_state = incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=target_padding_mask,
                    need_attn=False,
                    need_head_weights=False,
                )

            # 1 x B x C
            out_x = target_x[-1:,:,:]


        if self.layer_norm is not None:
            out_x = self.layer_norm(out_x)

        # T x B x C -> B x T x C
        out_x = out_x.transpose(0, 1)
        if self.project_out_dim is not None:
            out_x = self.project_out_dim(out_x)

        if self.output_hiddens:
            #num_layer x T x B x C  or  list[num_layer x T x B x C]
            out_hidden_states = hidden_states
        else:
            out_hidden_states = None

        return out_x, out_hidden_states


    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)





@register_model_architecture("transformer_rdangle_kv_sep", "transformer_rdangle_kv_sep")
def transformer_parsing(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.dangle_encoder_attention_heads = getattr(args, "dangle_encoder_attention_heads", 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.dis_encoder_layers = getattr(args, "dis_encoder_layers", 6)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.no_token_positional_embeddings = True
    args.use_rel_pos = True
    args.max_relative_position = getattr(args, "max_relative_position", 64)
    base_architecture(args)



