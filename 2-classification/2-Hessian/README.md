# Hessian Matrix

When critical point is at

* local minima: $H$ is **positive** definite = all eigen values are positive
* local maxima: $H$ is **negative** definitive = all eigen values are negative
* saddle point: some eigen values are positive and some are negative

However, in real scenario, gradient may not be accurately 0 and $H$ is complex.

As a result, we can assume some points are critical points like below rules.

$minimum\_ratio=\frac{number\_of\_positive\_eigen\_values}{number\_of\_eigen\_values}$

* local minima: `gradient norm(square) < 1e-3` and `minimum ratio > 0.5`
* saddle point: `gradient norm(square) < 1e-3` and `minimum ratio <= 0.5`