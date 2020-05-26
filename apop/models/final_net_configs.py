import torch
import pickle as pk
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
                      [('gru', {'input_size': 512, 'hidden_size': 256, 'num_layers': 2, 'batch_first': True, 'dropout': 0.25,
                                'bidirectional': True})]
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
    # n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout
    cnet_config = [6, 512, 64, 64, 8, 2048, 0.25, data['max_source_len'], len(data['idx_to_tokens'])]
  elif config == 'css_decoder':
    # output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device,
    # max_seq_len=100, score_fn, scaling_energy, multi_head, d_keys_values
    cnet_config = [len(data['idx_to_tokens']), 512, 1024, 6, 3, 0.25, 2, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                   data['max_source_len'], F.softmax, True, True, 64]
  elif config == 'multihead_objective_decoder':
    cnet_config = {'n_embeddings': len(data['idx_to_tokens']), 'emb_dim': 512, 'max_seq_len': data['max_source_len'],
                   'embedder_dropout': 0.25, 'd_model': 512, 'd_keys': 64, 'd_values': 64, 'n_heads': 8, 'mha_dropout': 0.25}
  else:
    cnet_config = None
  return cnet_config