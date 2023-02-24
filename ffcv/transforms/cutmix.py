"""
Mixup augmentation for images and labels (https://arxiv.org/abs/1710.09412)
"""
from typing import Tuple

from numba import objmode
import numpy as np
import torch as ch
import torch.nn.functional as F
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler



    # if args.beta > 0 and r < args.cutmix_prob:
    #         # generate mixed sample
    #         lam = np.random.beta(args.beta, args.beta)
    #         rand_index = torch.randperm(input.size()[0]).cuda()
    #         target_a = target
    #         target_b = target[rand_index]
    #         bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    #         input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    #         # adjust lambda to exactly match pixel ratio
    #         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            # output = model(input)
            # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

import numpy as np
from numba import types
from numba import TypingError
from numba.extending import overload

@overload(np.clip)
def impl_clip(a, a_min, a_max):
    # Check that `a_min` and `a_max` are scalars, and at most one of them is None.
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_min must be a_min scalar int/float")
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_max must be a_min scalar int/float")
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        raise TypingError("a_min and a_max can't both be None")

    if isinstance(a, (types.Integer, types.Float)):
        # `a` is a scalar with a valid type
        if isinstance(a_min, types.NoneType):
            # `a_min` is None
            def impl(a, a_min, a_max):
                return min(a, a_max)
        elif isinstance(a_max, types.NoneType):
            # `a_max` is None
            def impl(a, a_min, a_max):
                return max(a, a_min)
        else:
            # neither `a_min` or `a_max` are None
            def impl(a, a_min, a_max):
                return min(max(a, a_min), a_max)
    elif (
        isinstance(a, types.Array) and
        a.ndim == 1 and
        isinstance(a.dtype, (types.Integer, types.Float))
    ):
        # `a` is a 1D array of the proper type
        def impl(a, a_min, a_max):
            # Allocate an output array using standard numpy functions
            out = np.empty_like(a)
            # Iterate over `a`, calling `np.clip` on every element
            for i in range(a.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                out[i] = np.clip(a[i], a_min, a_max)
            return out
    else:
        raise TypingError("`a` must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    return impl


class ImageCutMix(Operation):
    """CutMix for images. Operates on raw arrays (not tensors).

    Parameters
    ----------
    alpha : float
        CutMix parameter alpha
    """

    def __init__(self, alpha: float, max_image_width: int, max_image_height: int, same_lambda: bool = True):
        super().__init__()
        self.alpha = alpha
        self.same_lambda = same_lambda
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

    def generate_code(self) -> Callable:
        alpha = self.alpha
        same_lam = self.same_lambda
        my_range = Compiler.get_iterator()
        max_image_width = self.max_image_width
        max_image_height = self.max_image_height 

        def mixer(images, dst, indices):
            def rand_bbox(W, H, lam):
                cut_rat = np.sqrt(1. - lam)
                cut_w = (W * cut_rat).astype(np.int64)
                cut_h = (H * cut_rat).astype(np.int64)

                # uniform
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                return bbx1, bby1, bbx2, bby2

            num_images, w, h, c = images.shape
            np.random.seed(indices[-1])
            lam = np.random.beta(alpha, alpha, (1,)) if same_lam else \
                  np.random.beta(alpha, alpha, num_images)
            bbx1, bby1, bbx2, bby2 = rand_bbox(max_image_width, max_image_height, lam)
            w_ratio = w / max_image_width
            h_ratio = h / max_image_height
            bbx1 = (bbx1 * w_ratio).astype(np.int64)
            bby1 = (bby1 * h_ratio).astype(np.int64)
            bbx2 = (bbx2 * w_ratio).astype(np.int64)
            bby2 = (bby2 * h_ratio).astype(np.int64)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            for ix in my_range(num_images):
                bbx1_, bby1_, bbx2_, bby2_ = (bbx1[0], bby1[0], bbx2[0], bby2[0]) if same_lam else \
                                             (bbx1[ix], bby1[ix], bbx2[ix], bby2[ix])
                dst[ix, :, :] = images[ix] 
                dst[ix, bbx1_:bbx2_, bby1_:bby2_] = images[ix -1, bbx1_:bbx2_, bby1_:bby2_]
            return dst

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (previous_state, AllocationQuery(shape=previous_state.shape,
                                                dtype=previous_state.dtype))

class LabelCutMix(Operation):
    """Mixup for labels. Should be initialized in exactly the same way as
    :cla:`ffcv.transforms.ImageCutMix`.
    """
    def __init__(self, alpha: float, max_image_width: int, max_image_height: int, same_lambda: bool = True):
        super().__init__()
        self.alpha = alpha
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height 
        self.same_lambda = same_lambda

    def generate_code(self) -> Callable:
        alpha = self.alpha
        same_lam = self.same_lambda
        my_range = Compiler.get_iterator()
        max_image_width = self.max_image_width
        max_image_height = self.max_image_height

        def mixer(labels, temp_array, indices):
            def rand_bbox(W, H, lam):
                cut_rat = np.sqrt(1. - lam)
                cut_w = (W * cut_rat).astype(np.int64)
                cut_h = (H * cut_rat).astype(np.int64)

                # uniform
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                return bbx1, bby1, bbx2, bby2
            num_labels = labels.shape[0]
            # permutation = np.random.permutation(num_labels)
            np.random.seed(indices[-1])
            lam = np.random.beta(alpha, alpha, (1,)) if same_lam else \
                  np.random.beta(alpha, alpha, num_labels)
            bbx1, bby1, bbx2, bby2 = rand_bbox(max_image_height, max_image_width, lam)

            for ix in my_range(num_labels):
                temp_array[ix, 0] = labels[ix][0]
                temp_array[ix, 1] = labels[ix - 1][0]
                temp_array[ix, 2] = bbx1[0] if same_lam else bbx1[ix]
                temp_array[ix, 3] = bby1[0] if same_lam else bby1[ix]
                temp_array[ix, 4] = bbx2[0] if same_lam else bbx2[ix]
                temp_array[ix, 5] = bby2[0] if same_lam else bby2[ix]

            return temp_array

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, shape=(6,), dtype=np.int64),
                AllocationQuery((6,), dtype=np.int64))