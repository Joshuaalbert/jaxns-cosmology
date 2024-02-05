import os
import sys

import jax
import pytest
from tf2jax._src.ops import register_operation, _check_attrs


@register_operation(op_name='Selu')
def selu(proto):
    _check_attrs(proto, {"T"})
    return jax.nn.selu


class MLModels:
    @classmethod
    def _content_path(cls):
        return os.path.split(os.path.abspath(sys.modules[cls.__module__].__file__))[:-1]

    def get_file(self, model_name: str):
        model_file = os.path.join(*self._content_path(), model_name, 'model.h5')
        if os.path.exists(model_file):
            return model_file
        else:
            raise FileNotFoundError(f"Model file {model_file} not found.")


def test_ml_models():
    ml_models = MLModels()
    with pytest.raises(FileNotFoundError):
        ml_models.get_file('mssm8')
    model_file = ml_models.get_file('mssm7')
    assert os.path.exists(model_file)
    model_file = ml_models.get_file('lcdm')
    assert os.path.exists(model_file)
