

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
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 1})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ]
                  ]
  elif config == 'attention_glu':
    cnet_config = [
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 2, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
                    ],
                    [
                      [('conv_block', {'in_chan': 512, 'out_chan': 1024, 'kernel': 3, 'stride': 1, 'pad': 1, 'dil': 1,
                                       'dropout': 0.1, 'groups': 1, 'k': 2})],
                      [('feed_forward', {'input_size': 512, 'output_size': 512, 'd_ff': 1024, 'dropout': 0.25})],
                      [('attention_conv_block', {'in_chan': 512, 'n_heads': 8, 'kernel': 5, 'dropout': 0., 'pad': 2, 'bias': True})]
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