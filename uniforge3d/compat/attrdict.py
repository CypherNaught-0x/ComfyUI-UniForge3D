"""Minimal replacement for ``easydict.EasyDict`` (recursive attribute access to dict keys)."""


class AttrDict(dict):
    def __init__(self, data=None, **kwargs):
        super().__init__()
        for key, value in {**(data or {}), **kwargs}.items():
            self[key] = value

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return AttrDict(value)
        if isinstance(value, (list, tuple)):
            return type(value)(AttrDict._wrap(item) for item in value)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value
