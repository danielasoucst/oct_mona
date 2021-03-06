# Copyright (C) 2013 Oskar Maier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.3.0
# since 2013-08-23
# status Release

# build-in modules

# third-party modules
import numpy
from scipy.ndimage.filters import gaussian_filter

# path changes


def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta / kappa) ** 2.) / float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1. / (1. + (delta / kappa) ** 2.) / float(spacing)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacong if not suppliec
    if None == voxelspacing:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in xrange(out.ndim)]

    for _ in xrange(niter):

        # calculate the diffs
        for i in xrange(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in xrange(out.ndim)]
            deltas[i][slicer] = numpy.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in xrange(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in xrange(out.ndim)]
            matrices[i][slicer] = numpy.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))

    return out