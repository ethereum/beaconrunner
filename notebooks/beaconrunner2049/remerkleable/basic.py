from typing import Any, TypeVar, Type
from remerkleable.core import BasicView, View

V = TypeVar('V', bound=View)


class boolean(int, BasicView):

    def encode_bytes(self) -> bytes:
        return b"\x01" if self else b"\x00"

    def __new__(cls, value: int):  # int value, but can be any subclass of int (bool, Bit, Bool, etc...)
        if value < 0 or value > 1:
            raise ValueError(f"value {value} out of bounds for bit")
        return super().__new__(cls, value)

    def __bool__(self):
        return self > 0

    @classmethod
    def coerce_view(cls: Type[V], v: Any) -> V:
        return cls(v)

    @classmethod
    def type_byte_length(cls) -> int:
        return 1

    @classmethod
    def decode_bytes(cls: Type[V], bytez: bytes) -> V:
        return cls(bytez != b"\x00")

    @classmethod
    def type_repr(cls) -> str:
        return "boolean"


class uint(int, BasicView):
    def __new__(cls, value: int):
        if value < 0:
            raise ValueError(f"unsigned type {cls} must not be negative")
        byte_len = cls.type_byte_length()
        if value.bit_length() > (byte_len << 3):
            raise ValueError(f"value out of bounds for {cls}")
        return super().__new__(cls, value)

    def __add__(self, other):
        return self.__class__(super().__add__(self.__class__.coerce_view(other)))

    def __sub__(self, other):
        return self.__class__(super().__sub__(self.__class__.coerce_view(other)))

    @classmethod
    def coerce_view(cls: Type[V], v: Any) -> V:
        if isinstance(v, uint) and cls.type_byte_length() != v.__class__.type_byte_length():
            raise ValueError("value must have equal byte length to coerce it")
        if isinstance(v, bytes):
            return cls.decode_bytes(v)
        return cls(v)

    @classmethod
    def decode_bytes(cls: Type[V], bytez: bytes) -> V:
        return cls(int.from_bytes(bytez, byteorder='little'))

    def encode_bytes(self) -> bytes:
        return self.to_bytes(length=self.__class__.type_byte_length(), byteorder='little')

    @classmethod
    def type_repr(cls) -> str:
        return f"uint{cls.type_byte_length()*8}"


class uint8(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 1


class uint16(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 2


class uint32(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 4


class uint64(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 8


class uint128(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 16


class uint256(uint):
    @classmethod
    def type_byte_length(cls) -> int:
        return 32


class bit(boolean):
    pass


class byte(uint8):
    pass
