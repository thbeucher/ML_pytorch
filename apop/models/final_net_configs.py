import torch
import pickle as pk
import torch.nn as nn
import torch.nn.functional as F


def get_encoder_config(config='base'):
  if config == 'separable':
    cnet_config = [
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                                  'dropout': 0.25, 'k': 1}),
                        ('separable_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                                  'dropout': 0.25, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'attention':
    cnet_config = [
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ]
                  ]
  elif config == 'attention_glu':
    cnet_config = [
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.25, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0.25, 'pad': 2, 'bias': True})]
                    ]
                  ]
  elif config == 'conv_attention':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'conv_attention_deep':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'conv_attention_deep2':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'conv_attention_deep3':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 16,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'conv_attention2':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 520, 'out_chan': 520, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 520, 'output_size': 520, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'conv_attention_large':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 544, 'kernel_conv': 5, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 544, 'kernel_conv': 5, 'stride_conv': 2, 'pad_conv': 4,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 544, 'kernel_conv': 5, 'stride_conv': 2, 'pad_conv': 6,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 544, 'output_size': 544, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 544, 'out_chan': 576, 'kernel_conv': 7, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 544, 'out_chan': 576, 'kernel_conv': 7, 'stride_conv': 1, 'pad_conv': 6,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 544, 'out_chan': 576, 'kernel_conv': 7, 'stride_conv': 1, 'pad_conv': 9,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 576, 'output_size': 576, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 576, 'out_chan': 608, 'kernel_conv': 9, 'stride_conv': 1, 'pad_conv': 4,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 576, 'out_chan': 608, 'kernel_conv': 9, 'stride_conv': 1, 'pad_conv': 8,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 576, 'out_chan': 608, 'kernel_conv': 9, 'stride_conv': 1, 'pad_conv': 12,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 608, 'output_size': 608, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 608, 'out_chan': 640, 'kernel_conv': 11, 'stride_conv': 1, 'pad_conv': 5,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 608, 'out_chan': 640, 'kernel_conv': 11, 'stride_conv': 1, 'pad_conv': 10,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 608, 'out_chan': 640, 'kernel_conv': 11, 'stride_conv': 1, 'pad_conv': 15,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 640, 'output_size': 640, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 640, 'out_chan': 672, 'kernel_conv': 15, 'stride_conv': 1, 'pad_conv': 7,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 640, 'out_chan': 672, 'kernel_conv': 15, 'stride_conv': 1, 'pad_conv': 14,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 640, 'out_chan': 672, 'kernel_conv': 15, 'stride_conv': 1, 'pad_conv': 21,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 672, 'output_size': 672, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  elif config == 'rnn_base':
    cnet_config = [
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})],
                      [('lstm', {'input_size': 512, 'hidden_size': 256, 'num_layers': 2, 'batch_first': True, 'dropout': 0.25,
                                 'bidirectional': True})]
                    ]
                  ]
  elif config == 'conv_transformer':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [('transformer', {'n_blocks': 4, 'd_model': 512, 'd_keys': 32, 'd_values': 32, 'n_heads': 16, 'd_ff': 2048,
                                        'dropout': 0.25, 'act_fn': 'relu', 'block_type': 'standard'})]
                    ]
                  ]
  elif config == 'conv_transformer_dec':
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 32,
                                                       'kernel_attn': 3, 'dropout_attn': 0.25, 'pad_attn': 1, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [('transformer_dec', {'n_blocks': 2, 'd_model': 512, 'd_keys': 32, 'd_values': 32, 'n_heads': 16, 'd_ff': 2048,
                                            'dropout': 0.25})]
                    ]
                  ]
  else:
    cnet_config = [
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 2, 'dil': 2,
                                        'dropout': 0.25, 'groups': 1, 'k': 1}),
                        ('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 3, 'dil': 3,
                                        'dropout': 0.25, 'groups': 1, 'k': 1})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  return cnet_config


def get_decoder_config(config='transformer', metadata_file='../ASR/_Data_metadata_letters_wav2vec.pk', **kwargs):
  with open(metadata_file, 'rb') as f:
    data = pk.load(f)

  if config == 'transformer':
    cnet_config = {'n_blocks': 6, 'd_model': 512, 'd_keys': 64, 'd_values': 64, 'n_heads': 8, 'd_ff': 2048, 'dropout': 0.25,
                   'max_seq_len': data['max_source_len'], 'output_dim': len(data['idx_to_tokens'])}
  elif config == 'css_decoder':
    cnet_config = {'output_dim': len(data['idx_to_tokens']), 'emb_dim': 512, 'hid_dim': 1024, 'n_layers': 6, 'kernel_size': 3,
                   'dropout': 0.25, 'pad_idx': data['tokens_to_idx']['<pad>'], 'max_seq_len': data['max_source_len'],
                   'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'score_fn': F.softmax,
                   'scaling_energy': True, 'multi_head': True, 'd_keys_values': 64}
  elif config == 'multihead_objective_decoder':
    cnet_config = {'n_embeddings': len(data['idx_to_tokens']), 'emb_dim': 512, 'max_seq_len': data['max_source_len'],
                   'embedder_dropout': 0.25, 'd_model': 512, 'd_keys': 64, 'd_values': 64, 'n_heads': 8, 'mha_dropout': 0.25}
  else:
    cnet_config = None
  return cnet_config


def get_input_proj_layer(config='base'):
  if config == 'base2':
    input_proj = nn.Sequential(nn.Dropout(0.25),
                               nn.Linear(520, 520),
                               nn.ReLU(inplace=True),
                               nn.LayerNorm(520))
  else:
    input_proj = nn.Sequential(nn.Dropout(0.25),
                               nn.Linear(512, 512),
                               nn.ReLU(inplace=True),
                               nn.LayerNorm(512))
  return input_proj