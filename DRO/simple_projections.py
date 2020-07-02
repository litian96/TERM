"""simple_projections.py
A small module for solving a distributionally robust optimization on
the chi-square divergence ball on the simplex. The bisection algorithm
takes a desired (relative) solution tolerance as a parameter, and is
extremely quick for reasonable values of n and tolerance.
 Given a n-dimensional vector w and a positive number rho, solves
 minimize_p   .5 * norm(p - w, 2)^2
   s.t.      sum(p) = 1, p >= 0,
             (1/nn) * .5 * sum_{i=1}^n (n * p[i] - 1)^2  <=  rho.
"""

## code from: https://github.com/hsnamkoong/robustopt/blob/master/src/simple_projections.py

import numpy as np
import numpy.matlib
import math

"""
 p = project_onto_chi_square_ball(w, rho, tol = 1e-10)
 Solves the projection problem given above by bisecting on the dual problem
 maximize_{lam >= 0}  min_{p}  .5 * norm(p - w, 2)^2 
                                 - lam * (rho + .5 - .5 * n * norm(p)^2)
                        s.t.    sum(p) = 1, p >= 0
 where we used (1/nn) * .5 * sum_{i=1}^n (n * p[i] - 1)^2 = .5 * (n * norm(p)^2 - 1)
 and duality. The KKT conditions of the inner minimization problem are given by
 p(lam) = (1 / (1 + lam * n)) * max(w - eta, 0)
 where eta is the dual variable for the constraint sum(p) = 1. We
 solve eta such that sum(p(lam)) = 1 in solve_inner_eta.
 Given such eta, first note that the gradient of the dual objective 
 g(lam) = min_{p}  .5 * norm(p - w, 2)^2 
                     - lam * (rho + .5 - .5 * n * norm(p)^2)
            s.t.    sum(p) = 1, p >= 0
 with respect to lam is given by

 g'(lam) = - (rho + .5 - .5 * n * norm(p(lam))^2).
 Since g is concave, g' is decreasing in lam. Hence, we bisect to find
 the optimal lam:
 If g'(lam) > 0,  i.e. .5 * n * norm(p(lam))^2 > rho + .5,  increase lam
 If g'(lam) < 0,  i.e. .5 * n * norm(p(lam))^2 < rho + .5,  decrease lam.
 -------------- obtaining an finite upper bound for lam^* ---------------
 Note that the optimal dual solution lam^* satisfies
 .5 * n * (1/ (1 + lam * n)^2) * sum (w[i] - eta)^2 = rho + .5
 so that this gives th bound
 (1 + lam^* * n)^2 <= maximum(w)^2 * n^2 / (2 * rho + 1)
 or equivalently,
 lam^* <= (1/n) * (n * maximum(w) / sqrt(2 * rho + 1) - 1) := lam_max.
 ------------------------------------------------------------------------
"""


def project_onto_chi_square_ball(w, rho, tol=1e-10):
    assert (rho > 0)
    rho = float(rho)


    # sort in decreasing order
    w_sort = np.sort(w)  # increasing
    w_sort = w_sort[::-1]  # decreasing

    w_sort_cumsum = w_sort.cumsum()
    w_sort_sqr_cumsum = np.square(w_sort).cumsum()
    nn = float(w_sort.shape[0])

    lam_min = 0.0
    lam_max = (1 / nn) * (nn * w_sort[0] / np.sqrt(2. * rho + 1.) - 1.)
    lam_init_max = lam_max

    if (lam_max <= 0):  # optimal lambda is 0
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, 0., rho)
        p = w - eta
        low_inds = p < 0
        p[low_inds] = 0.
        return p

    # bisect on lambda to find the optimal lambda value
    while (lam_max - lam_min > tol * lam_init_max):
        lam = .5 * (lam_max + lam_min)
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)

        # compute norm(p(lam))^2 * (1+lam * nn)^2
        thresh = .5 * nn * (w_sort_sqr_cumsum[ind] - 2. * eta * w_sort_cumsum[ind] + eta ** 2 * (ind + 1.))
        if (thresh > (rho + .5) * (1 + lam * nn) ** 2):
            # constraint infeasible, increase lam (dual var)
            lam_min = lam
        else:
            # constraint loose, decrease lam (dual var)
            lam_max = lam

    lam = .5 * (lam_max + lam_min)
    (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
    p = w - eta
    low_inds = p < 0
    p[low_inds] = 0
    return (1. / (1. + lam * nn)) * p


"""solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
 Given lam, solves the optimization problem 
 minimize_{p}  .5 * norm(p - w, 2)^2 
                     - lam * (rho + .5 - .5 * n * norm(p)^2)
     s.t.      sum(p) = 1, p >= 0
 by solving for eta that satifies sum(p(lam)) = 1 where 
 p(lam) = (1 / (1 + lam * n)) * max(w - eta, 0).
 Here, eta is the dual variable for sum(p) = 1. Let w_sort be a sorted
 version of w in decreasing order. Plugging the above equation into
 sum(p) = 1, we obtain
 eta = (1/I) (sum_{i=1}^I w_sort[i] - (1 + lam * n))   ...  (*)
 where I = max{i: w_sort[i] >= eta}. Hence, it suffices to solve for
 I to solve for eta. To this end, define
 f(j) = w_sort[j] - (1/j) * (sum_{i=1}^j w_sort[i] - (1 + lam * n)).
 Then, we have that I = max{j : f(j) >= 0} from which eta can be
 computed as in (*). The function returns the tuple (eta, I).
"""


def solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho):
    fs = w_sort - (w_sort_cumsum - (1. + lam * nn)) / (np.arange(nn) + 1.)
    ind = (fs > 0).sum() - 1
    return ((1 / (ind + 1.)) * (w_sort_cumsum[ind] - (1. + lam * nn)), ind)


def test_projections(tol=1e-5):
    # Desired values are somewhat inaccurate since numerical accuracy
    # was lost when copying w from julia to python. Gurobi can also
    # slightly inaccurate than these specialized solvers.

    num_passed = 0

    w1 = np.array([0.652694,
                   0.104329,
                   0.741784,
                   -0.886538,
                   1.704,
                   -0.460611,
                   1.38062,
                   0.0115509,
                   -0.522331,
                   -0.428545,
                   0.57985,
                   0.238284,
                   2.36878,
                   -0.100438,
                   -0.0603701,
                   0.429369,
                   -0.95219,
                   0.813954,
                   0.885749,
                   0.179369,
                   0.288465,
                   -1.32917,
                   3.84637,
                   0.837628,
                   -0.325304,
                   -0.331649,
                   2.16613,
                   -0.712114,
                   -0.971334,
                   -0.639229])

    p1_desired = np.array([0.0401005,
                           0.0124487,
                           0.0446238,
                           3.44244e-9,
                           0.0934907,
                           1.95506e-8,
                           0.077079,
                           0.00767349,
                           7.61686e-9,
                           4.14356e-8,
                           0.0364103,
                           0.0192461,
                           0.12722,
                           0.00202305,
                           0.003972,
                           0.0288177,
                           3.02299e-9,
                           0.048292,
                           0.0519424,
                           0.0162738,
                           0.0217614,
                           1.87036e-9,
                           0.20219,
                           0.0494957,
                           4.13615e-7,
                           3.58341e-7,
                           0.116939,
                           5.34551e-9,
                           2.92164e-9,
                           5.89345e-9])

    p2_desired = np.array([6.85178e-7,
                           -5.90154e-7,
                           3.04519e-7,
                           -1.04984e-6,
                           1.84985e-6,
                           -9.72425e-7,
                           -2.07198e-7,
                           -7.16248e-7,
                           -9.86864e-7,
                           -9.64289e-7,
                           7.0556e-7,
                           -3.1829e-7,
                           0.135177,
                           -8.19966e-7,
                           -7.87841e-7,
                           2.95476e-7,
                           -1.05675e-6,
                           -3.07837e-7,
                           -7.02439e-7,
                           -4.52646e-7,
                           -1.84521e-7,
                           -1.0411e-6,
                           0.824686,
                           -4.9341e-7,
                           -9.33656e-7,
                           -9.35794e-7,
                           0.0401506,
                           -1.02412e-6,
                           -1.05841e-6,
                           -1.01091e-6])

    nn = 30
    rho = 1.
    p1 = project_onto_chi_square_ball(w1, rho)

    kkt_error = abs(rho + .5 - .5 * nn * np.linalg.norm(p1) ** 2) + abs(np.sum(p1) - 1.0) + abs(np.sum(p1[p1 < 0]))

    if kkt_error > tol:
        print("Test failed for nn=" + str(nn) + " and rho=" + str(rho) + " with error=" + str(
            np.linalg.norm(p1_desired - p1, 1)) + " and kkt_error = " + str(kkt_error))
    else:
        num_passed += 1

    rho = 10.
    p2 = project_onto_chi_square_ball(w1, rho)

    kkt_error = abs(rho + .5 - .5 * nn * np.linalg.norm(p2) ** 2) + abs(np.sum(p2) - 1.0) + abs(np.sum(p2[p2 < 0]))

    if kkt_error > tol:
        print("Test failed for nn=" + str(nn) + " and rho=" + str(rho) + " with error=" + str(
            np.linalg.norm(p2_desired - p2, 1)) + " and kkt_error = " + str(kkt_error))
    else:
        num_passed += 1

    print("Passed " + str(num_passed) + " out of 2 tests")


def main():
    test_projections(1e-4)


if __name__ == '__main__':
    main()