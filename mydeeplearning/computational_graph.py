class ComputationalGraphNode:
    """計算グラフノード"""

    def forward(self, *args, **kwargs):
        """順伝播"""
        raise NotImplementedError()

    def backward(self, dout):
        """逆伝播"""
        raise NotImplementedError()


class MulNode(ComputationalGraphNode):
    """乗算ノード"""

    def forward(self, x, y):
        self._x = x
        self._y = y
        return x * y

    def backward(self, dout):
        return dout * self._y, dout * self._x


class AddNode(ComputationalGraphNode):
    """加算ノード"""

    def forward(self, x, y):
        self._x = x
        self._y = y
        return x + y

    def backward(self, dout):
        return dout * 1, dout * 1
