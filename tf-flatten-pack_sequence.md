---
layout: default
---

## What is TensorFlow tf.nest.pack_sequence_as and tf.nest.flatten and where should I use it?

_I came across these methods while digging through tf-privacy. They use flatten right before calculating the L2-norms to clip the gradients as part of the clipping and noising proposed by DP-SGD._

Here are a few examples for how flatten and pack_sequence_as can be used in conjunction:

```
a = [1, [2, 3], [[4, 5], [6, 7]], [[[8, 9], [10, 11]]]]
b = {1: 1, 2: 2, 3: {4: 4, 5: 5}, 6: [[7, 8], [9, 10]]}

from tensorflow.python.util import nest

nest.flatten(a)

# output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

nest.flatten(b)

# output: [1, 2, 4, 5, 7, 8, 9, 10]

a = [100, 200, 300, 400, 500, 600, 700, 800]
b = {1: 1, 2: 2, 3: {4: 4, 5: 5}, 6: [[7, 8], [9, 10]]}
c = nest.pack_sequence_as(structure=b , flat_sequence=a)

# output: c: {1: 100, 2: 200, 3: {4: 300, 5: 400}, 6: [[500, 600], [700, 800]]}
```

[back](./)
