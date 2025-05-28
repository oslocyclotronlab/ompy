
from typing import Iterator, Any
from dataclasses import dataclass


class FwfParseError(Exception):
    pass


def parse_fwf(string: str, fmt: str, space_as_delim: bool = False,
              convert_missing: bool = False, missing_value: Any = None) -> Iterator[Any]:
    """ Parse a Fortran fixed-width formatted string

    Doesn't support nested formats or any other fancy stuff.

    Parameters
    ----------
    string : str
        Line to parse. Newstrings are ignored.
    fmt : str
        Format string. This is a string of integers separated by commas
        indicating the width of each field in the string.
    space_as_delim : bool, optional
        If True, the fields are separated by 1 space.
    convert_missing : bool, optional
        If True, convert missing values to the `missing_value`.
    missing_value : Any, optional
        The value to convert missing values to.

    Returns
    -------
    Iterator[Any]
        An iterator of the parsed fields.

    Raises
    ------
    ValueError
        If the sum of the field widths does not match the length of the string.

    Examples
    --------
    >>> list(parse_fwf(" 12  fish 3.14", "i3, a6, f4.2"))
    (12, "fish", 3.14)
    >>> list(parse_fwf("  5 4 3 2112341.423   21.300", "5i2, 2f7.3"))
    (5, 4, 3, 2, 11, 2341.423, 21.3)
    """
    # Ignore initial parentheses
    fmt = fmt.lstrip("(").rstrip(")")
    fields = parse_fwf_format(fmt)
    scale = 0.0
    for i, field in enumerate(fields):
        try:
            scale = field.set_scale(scale)
            value, string = field.parse(string, scale=scale)
        except ValueError as e:
            # When the parsing fails, the string is not updated
            _head, _body = field.split(string)
            if convert_missing:
                yield missing_value
                string = _body
                continue
            else:
                raise FwfParseError(f"Error converting \"{_head}\" using {field}:\n"
                                    f"{highlight_error(string, 0, field.width)}") from e
        except Exception as e:
            raise FwfParseError(f"Error parsing field {i}: {field} of string `{string}`") from e

        if string and space_as_delim:
            string = string[1:]
            
        match field:
            case Space() | Scale(spec=None):
                continue
            case _:
                yield value


def highlight_error(input_string, error_position, error_length=1):
    """
    Constructs an error message with the faulty part of the input string underlined.

    Parameters:
    - input_string: The original input string where the error occurred.
    - error_position: The index of the start of the error in the input string.
    - error_length: The length of the erroneous part. Defaults to 1.

    Returns:
    A string combining the original input and an underline indicating the error location.
    """
    pointer_line = ' ' * error_position + '^' * max(error_length, 1)
    truncated = (input_string[:70 - 3] + '...') if len(input_string) > 70 else input_string
    truncated = truncated.replace(" ", "␣")
    error_message = f"{truncated}\n{pointer_line}\n"
    return error_message

def split_string_fwf(string: str, fmt: str, space_as_delim: bool = False) -> tuple:
    """ Split a string according to a Fortran fixed-width format

    Parameters
    ----------
    string : str
        String to split
    fmt : str
        Format string. This is a string of integers separated by commas
        indicating the width of each field in the line.
    space_as_delim : bool, optional
        If True, the fields are separated by 1 space.

    Returns
    -------
    tuple
        A tuple of the parsed fields.

    Raises
    ------
    ValueError
        If the sum of the field widths does not match the length of the line.

    Examples
    --------
    >>> split_string_fwf(" 12  fish 3.14", "i3, a6, f4.2")
    (12, "fish", 3.14)
    >>> split_string_fwf("  5 4 3 2112341.423   21.300", "5i2, 2f7.3")
    (5, 4, 3, 2, 11, 2341.423, 21.3)
    """
    fields = parse_fwf_format(fmt)
    for i, field in enumerate(fields):
        try:
            head, string = field.split(string)
        except Exception as e:
            raise FwfParseError(f"Error parsing field {i}: {field} of string `{string}`") from e

        if space_as_delim:
            yield ' '

        yield head



@dataclass(kw_only=True)
class Field:
    type: type
    width: int

    def split(self, string: str) -> tuple[str, str]:
        return string[:self.width], string[self.width:]

    def parse(self, string: str, scale: float = 1.0) -> tuple[Any, str]:
        head, body = self.split(string)
        return self.type(head), body

    def set_scale(self, scale: float) -> float:
        """ The reader should call this method to set the scale 
        Only the p-format field should set the scale
        """
        return scale

@dataclass(kw_only=True)
class NumericField(Field):
    def parse(self, string: str, scale: float = 0.0) -> tuple[Any, str]:
        head, body = self.split(string)
        factor = 10**scale
        return self.type(self.type(head) * factor), body

@dataclass(kw_only=True)
class IntField(NumericField):
    type: type = int


@dataclass(kw_only=True)
class FloatField(NumericField):
    type: type = float
    decimals: int


@dataclass(kw_only=True)
class CharField(Field):
    type: type = str


@dataclass(kw_only=True)
class Space(Field):
    type: type = str

    def split(self, string: str) -> tuple[str, str]:
        return '', string[self.width:]


@dataclass(kw_only=True)
class Scientific(NumericField):
    type: type = float
    exponent: int


@dataclass(kw_only=True)
class Scale(Field):
    type: type = float  # Not needed
    scale: float
    spec: Field | None  # None means 0p, reset scaling

    def parse(self, string: str, scale: float = 0.0) -> tuple[Any, str]:
        # We ignore the given scale
        if self.spec is None:
            return '', string
        return self.spec.parse(string, self.scale)

    def set_scale(self, scale: float) -> float:
        return self.scale


def parse_fwf_format(fmt: str) -> Iterator[Field]:
    """ Parse a Fortran fixed-width format string

    Parameters
    ----------
    fmt : str
        Format string. This is a string of integers separated by commas
        indicating the width of each field in the line.

    Returns
    -------
    Iterator[Field]
        An iterator of the parsed fields.

    Examples
    --------
    >>> list(parse_fwf_format("i2, a6, f2.2"))
    [IntField(type=<class 'int'>, width=2), CharField(type=<class 'str'>, width=6), FloatField(type=<class 'float'>, width=2, decimals=2)]
    >>> list(parse_fwf_format("3i2, 2f4.3"))
    [IntField(type=<class 'int'>, width=2), IntField(type=<class 'int'>, width=2), IntField(type=<class 'int'>, width=2), FloatField(type=<class 'float'>, width=4, decimals=3), FloatField(type=<class 'float'>, width=4, decimals=3)]
    """
    for field in fmt.split(","):
        repeats, type_fmt, body = split_field_spec(field.strip())
        match type_fmt:
            case "i":
                spec = IntField(width=int(body))
            case "f":
                width, decimals = map(int, body.split("."))
                spec = FloatField(width=width, decimals=decimals)
            case "a":
                spec = CharField(width=int(body))
            case 'x':
                # We compress the spaces into a single space
                width = 1 if not body else int(body)
                spec = Space(width=repeats*width)
                repeats = 1
            case 'e':
                width, exponent = map(int, body.split("."))
                spec = Scientific(width=width, exponent=exponent)
            case 'p':
                # For the p-format, the initial number is the scale,
                # not the number of repeats
                scale = repeats
                repeats = 1
                if not body:
                    # Reset scaling
                    spec = Scale(scale=0, spec=None, width=0)
                else:
                    # We assume the nested format is a valid format string
                    nested_specs = list(parse_fwf_format(body))
                    # We can only have one nested spec
                    if len(nested_specs) != 1:
                        raise ValueError(f"Invalid nested format {body}")
                    nested_spec = nested_specs[0]
                    spec = Scale(scale=float(scale), spec=nested_spec,
                                 width=nested_spec.width)
            case _:
                raise ValueError(f"Invalid field type {type_fmt}")

        for _ in range(repeats):
            yield spec


def split_field_spec(spec: str) -> tuple[int, str, str]:
    """ Split a field specification into width and type

    Parameters
    ----------
    spec : str
        A field specification. This is a string of the form "i2", "f4.2", etc.

    Returns
    -------
    tuple
        A tuple of the repeats, field type, and rest

    Examples
    --------
    >>> split_field_spec("i2")
    (1, 'i', '2')
    >>> split_field_spec("f4.2")
    (1, 'f', '4.2')
    >>> split_field_spec("3i2")
    (3, 'i', '2')
    """
    head = ''
    body = spec
    while body and body[0].isdigit():
        head += body[0]
        body = body[1:]
    if not head:
        head = 1
    return int(head), body[0], body[1:]
