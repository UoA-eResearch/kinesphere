"""
Load a PyTorch checkpoint without running arbitrary pickled code.

Bailando's checkpoints store their config as an easydict.EasyDict next to the weights, which
PyTorch's weights_only loader refuses. This unpickler allows only what a checkpoint of tensors
needs (tensor and storage rebuilders, OrderedDict, a few builtins) and turns EasyDict into a plain
dict subclass; any other class or function in the file raises an error instead of being imported.
"""
import pickle
import types

import torch


class _Dict(dict):
    """Stands in for easydict.EasyDict (a dict that also keeps attributes)."""


_ALLOWED = {
    ('collections', 'OrderedDict'),
    ('torch._utils', '_rebuild_tensor'),
    ('torch._utils', '_rebuild_tensor_v2'),
    ('torch._utils', '_rebuild_tensor_v3'),
    ('torch._utils', '_rebuild_parameter'),
    ('torch._utils', '_rebuild_parameter_with_state'),
    ('torch._tensor', '_rebuild_from_type_v2'),
    ('torch', 'Size'),
    ('torch', 'Tensor'),
    ('torch.nn.parameter', 'Parameter'),
    ('builtins', 'set'),
    ('builtins', 'frozenset'),
    ('builtins', 'slice'),
    ('builtins', 'complex'),
}
_STORAGES = {'FloatStorage', 'DoubleStorage', 'HalfStorage', 'BFloat16Storage', 'LongStorage', 'IntStorage',
             'ShortStorage', 'CharStorage', 'ByteStorage', 'BoolStorage', 'UntypedStorage'}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('easydict', 'EasyDict'):
            return _Dict
        if (module, name) in _ALLOWED or (module == 'torch' and name in _STORAGES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing to load {module}.{name} from a checkpoint')


_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__='safe_pickle')


def load_checkpoint(path):
    """The checkpoint's contents (for Bailando: {'model': state_dict, 'config': ..., 'epoch': ...})."""
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_module)
