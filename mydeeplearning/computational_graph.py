class MulNode:
    """乗算ノード"""

    def forward(self, x, y):
        """順伝播"""
        self._x = x
        self._y = y
        return x * y

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
        return dout * self._y, dout * self._x
