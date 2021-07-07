"""Numerical approximations."""


class ApproximateMixin:
    def ladder_ptransp(self, x, y, v, method, n_steps):
        """Perform an approximate parallel transport.

        Marco, Lorenzi, and Xavier Pennec. "Parallel transport with Pole
        ladder: Application to deformations of time series of images."
        International Conference on Geometric Science of
        Information. Springer, Berlin, Heidelberg, 2013.

        Args:
          method: either "pole" or "schild" transport algorithm
          n_steps: number of iterations
        """
        if not method in ["pole", "schild"]:
            raise ValueError("Invalid transport method {}".format(method))
        if n_steps <= 0:
            raise ValueError("n_steps should be greater than zero")
        u = self.log(x, y)
        v_i = v / n_steps
        x_i = self.exp(x, v_i)
        x_prev = x
        for i in range(1, n_steps + 1):
            u_i = u * i / n_steps
            y_i = self.exp(x, u_i)
            if method == "pole":
                u_1 = self.log(x_prev, y_i) / 2.0
                half_geo = self.exp(x_prev, u_1)
                u_2 = -self.log(half_geo, x_i)
                end_geo = self.exp(half_geo, u_2)
                v_i = -self.log(y_i, end_geo)
                x_i = self.exp(y_i, v_i)
            elif method == "schild":
                u_1 = self.log(x_i, y_i) / 2.0
                half_geo = self.exp(x_i, u_1)
                u_2 = -self.log(half_geo, x_prev)
                end_geo = self.exp(half_geo, u_2)
                v_i = self.log(y_i, end_geo)
                x_i = end_geo
            x_prev = y_i
        return v_i * n_steps
