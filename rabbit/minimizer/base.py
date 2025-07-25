import tensorflow as tf
from scipy.optimize import OptimizeResult
from .function import ScalarFunction
from abc import ABC, abstractmethod

# Dummy status messages
status_messages = (
    "Optimization terminated successfully.",
    "Maximum number of iterations exceeded.",
    "A bad approximation caused failure to predict improvement.",
    "A linalg error occurred, such as a non-psd Hessian.",
)


class BaseQuadraticSubproblem(ABC):
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method and
    ``hess_prod`` property.
    """
    def __init__(self, x, closure):
        # evaluate closure
        f, g, hessp, hess = closure(x)

        self._x = x
        self._f = f
        self._g = g
        self._h = hessp if self.hess_prod else hess
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None

    def __call__(self, p):
        return self.fun + tf.tensordot(self.jac, p, axes=1) + 0.5 * tf.tensordot(p, self.hessp(p), axes=1)

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self.hess_prod:
            raise Exception('class {} does not have '
                            'method `hess`'.format(type(self)))
        return self._h

    def hessp(self, p):
        """Value of Hessian-vector product at current iteration for a
        particular vector ``p``.

        Note: ``self._h`` is either a Tensor or a LinearOperator. In either
        case, it has a method ``mv()``.
        """
        return tf.linalg.matvec(self._h, p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = tf.norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = tf.tensordot(d,d, axes=1)
        b = 2 * tf.tensordot(z,d, axes=1)
        c = tf.tensordot(z,z, axes=1) - trust_radius**2
        sqrt_discriminant = tf.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically equivalent to:
        #   ta = (-b - sqrt_discriminant) / (2*a)
        #   tb = (-b + sqrt_discriminant) / (2*a)
        # but produces smaller round off errors.
        aux = b + sqrt_discriminant * tf.sign(b)
        tab = tf.stack([-aux / (2 * a), -2 * c / aux])
        return tf.sort(tab)

    @abstractmethod
    def solve(self, trust_radius):
        pass

    @property
    @abstractmethod
    def hess_prod(self):
        """A property that must be set by every sub-class indicating whether
        to use full hessian matrix or hessian-vector products."""
        pass


def _minimize_trust_region(fun, x, subproblem=None, initial_trust_radius=1.,
                           max_trust_radius=1000., eta=0.15, gtol=1e-4,
                           max_iter=None, disp=False, return_all=False,
                           callback=None):

    if subproblem is None:
        raise ValueError("A subproblem solving strategy is required for trust-region methods")
    if not (0 <= eta < 0.25):
        raise Exception("Invalid acceptance stringency")
    if max_trust_radius <= 0:
        raise Exception("Max trust radius must be positive")
    if initial_trust_radius <= 0:
        raise ValueError("Initial trust radius must be positive")
    if initial_trust_radius >= max_trust_radius:
        raise ValueError("Initial trust radius must be less than max")

    disp = int(disp)
    if max_iter is None:
        max_iter = tf.size(x).numpy() * 200

    # Prepare scalar function object (user must define closure)
    hessp = subproblem.hess_prod
    sf = ScalarFunction(fun, hessp=hessp, hess=not hessp)
    closure = sf.closure

    # warnflag = 1
    k = 0

    trust_radius = tf.convert_to_tensor(initial_trust_radius, dtype=x.dtype)
    # x = tf.reshape(tf.identity(x0), [-1])
    if return_all:
        allvecs = [x]

    m = subproblem(x, closure)

    while k < max_iter:
        try:
            p, hits_boundary = m.solve(trust_radius)
        except tf.errors.InvalidArgumentError as exc:
            warnflag = 3
            break

        predicted_value = m(p)
        x_proposed = x + p
        # m_proposed = fun(x_proposed) # subproblem(x_proposed, closure)

        actual_reduction = m.fun - fun(x_proposed)
        predicted_reduction = m.fun - predicted_value

        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        if rho < 0.25:
            trust_radius = trust_radius * 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = tf.minimum(2 * trust_radius, max_trust_radius)

        if rho > eta:
            x.assign(x_proposed)
            # m = m_proposed
            m = subproblem(x, closure)
        # else:#if isinstance(sf, Minimizer):  # user-defined class

        if return_all:
            allvecs.append(tf.identity(x))
        if callback is not None:
            callback(tf.identity(x))
        k += 1

        # if disp > 1:
        print("iter", k, "fval:", m.fun)

        if m.jac_mag < gtol:
            warnflag = 0
            break

    return sf.time_closure, sf.n_closure

    # if disp:
    #     msg = status_messages[warnflag]
    #     if warnflag != 0:
    #         msg = "Warning: " + msg
    #     tf.print(msg)
    #     tf.print("Current function value:", m.fun)
    #     tf.print("Iterations:", k)
    #     tf.print("Function evaluations:", sf.nfev)

    # result = OptimizeResult(
    #     x=tf.reshape(x, x.shape).numpy(),
    #     fun=m.fun.numpy(),
    #     grad=tf.reshape(m.jac, x.shape).numpy(),
    #     success=(warnflag == 0),
    #     status=warnflag,
    #     nfev=sf.nfev,
    #     nit=k,
    #     message=status_messages[warnflag]
    # )

    # if not subproblem.hess_prod:
    #     result["hess"] = tf.reshape(m.hess, [*x.shape, *x.shape]).numpy()

    # if return_all:
    #     result["allvecs"] = [v.numpy() for v in allvecs]

    # return result