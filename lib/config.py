
import re

class Config(dict):

  IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

  def __init__(self, *args, **kwargs):
    self._data = dict(*args, **kwargs)
    super().__init__(self._data)

  def __getattr__(self, name):
    if name.startswith('_'):
      return super().__getattr__(name)
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, key, value):
    if key.startswith('_'):
      return super().__setattr__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def __getitem__(self, name):
    try:
      result = self._data[name]
    except TypeError:
      raise KeyError(f"When updating the config, make sure you have this key {name} already in the default config")
    return result

  def __reduce__(self):
    return (type(self), (dict(self),))

  def __setitem__(self, key, value):
    if key.startswith('_'):
      return super().__setitem__(key, value)
    message = f"Tried to set key '{key}' on immutable config. Use update()."
    raise AttributeError(message)

  def update(self, *args, **kwargs):
    result = self._data.copy()
    inputs = dict(*args, **kwargs)
    for key, new in inputs.items():
      if self.IS_PATTERN.match(key):
        pattern = re.compile(key)
        keys = {k for k in result if pattern.match(k)}
      else:
        keys = [key]
      if not keys:
        raise KeyError(f'Unknown key or pattern {key}.')
      for key in keys:
        old = result[key]
        try:
          if isinstance(old, int) and isinstance(new, float):
            if float(int(new)) != new:
              message = f"Cannot convert fractional float {new} to int."
              raise ValueError(message)
          result[key] = type(old)(new)
        except (ValueError, TypeError):
          raise TypeError(
              f"Cannot convert '{new}' to type '{type(old).__name__}' " +
              f"for key '{key}' with previous value '{old}'.")
    return type(self)(result)

  def __str__(self):
    lines = ['\nConfig:']
    keys, vals, typs = [], [], []
    for key, val in self.items():
      keys.append(key + ':')
      vals.append(str(val))
      typs.append(str(type(val).__name__))
    max_key = max(len(k) for k in keys) if keys else 0
    max_val = max(len(v) for v in vals) if vals else 0
    for key, val, typ in zip(keys, vals, typs):
      key = key.ljust(max_key)
      val = val.ljust(max_val)
      lines.append(f'{key}  {val}  ({typ})')
    return '\n'.join(lines)


## Example
# config = Config(
#   pop_size = 10,
#   dim = 2,
#   r=3.0
# )
# config = config.update(r=2.0)
# print(config)
