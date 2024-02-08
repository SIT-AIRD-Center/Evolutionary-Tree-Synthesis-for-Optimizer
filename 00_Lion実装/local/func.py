import tensorflow as tf

class BaseFunc:
    # 関数
    function = None
    # 関数の引数の数
    num_args = None
    def __init__(self):
        pass

    # 演算する関数
    def __call__(self, X):
        # 引数の数が合っているか確認
        if len(X) != self.num_args: raise

        # 演算
        result = self.function(*X)

        # 演算結果にNanやInfが含まれている場合，0に置き換える
        result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
        result = tf.where(tf.math.is_inf(result), tf.zeros_like(result), result)

        return result

class AddFunc(BaseFunc):
    def __init__(self):
        self.function = tf.add
        self.num_args = 2

class SubFunc(BaseFunc):
    def __init__(self):
        self.function = tf.subtract
        self.num_args = 2

class MulFunc(BaseFunc):
    def __init__(self):
        self.function = tf.multiply
        self.num_args = 2

class DivFunc(BaseFunc):
    def __init__(self):
        self.function = tf.divide
        self.num_args = 2

class LogFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.log
        self.num_args = 1

class Log1pFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.log1p
        self.num_args = 1

class SinFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.sin
        self.num_args = 1

class CosFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.cos
        self.num_args = 1

class TanFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.tan
        self.num_args = 1

class SinhFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.sinh
        self.num_args = 1

class CoshFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.cosh
        self.num_args = 1

class TanhFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.tanh
        self.num_args = 1

class SquareFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.square
        self.num_args = 1

class SqrtFunc(BaseFunc):
    def __init__(self):
        self.function = tf.sqrt
        self.num_args = 1

class PowFunc(BaseFunc):
    def __init__(self):
        self.function = tf.pow
        self.num_args = 2

class ExpFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.exp
        self.num_args = 1

class AbsFunc(BaseFunc):
    def __init__(self):
        self.function = tf.abs
        self.num_args = 1

class ArcsinFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.asin
        self.num_args = 1

class ArccosFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.acos
        self.num_args = 1

class ArctanFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.atan
        self.num_args = 1

class ArcsinhFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.asinh
        self.num_args = 1

class ArccoshFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.acosh
        self.num_args = 1

class ArctanhFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.atanh
        self.num_args = 1

class SignFunc(BaseFunc):
    def __init__(self):
        self.function = tf.math.sign
        self.num_args = 1

class MaxFunc(BaseFunc):
    def __init__(self):
        self.function = tf.maximum
        self.num_args = 2