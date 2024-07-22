class ComputationalGraphNode:
    """計算グラフノード"""

    def forward(self, x, y):
        """順伝播"""
        self._x = x
        self._y = y
        return self._compute(x, y)

    def _compute(self, x, y):
        """順伝播の計算を行う"""
        raise NotImplementedError()

    def backward(self, dout):
        """逆伝播

        Parameters
        ----------
        dout : any
            逆伝搬の上流から伝わってきた微分

        Returns
        -------
        any, any
            ノードへの順伝播の2つの入力の微分
        """
        raise NotImplementedError()


class MulNode(ComputationalGraphNode):
    """乗算ノード"""

    def _compute(self, x, y):
        return x * y

    def backward(self, dout):
        return dout * self._y, dout * self._x


class AddNode(ComputationalGraphNode):
    """加算ノード"""

    def _compute(self, x, y):
        return x + y

    def backward(self, dout):
        return dout * 1, dout * 1
