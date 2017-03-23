import numpy as np

def pad_size(size_in, min_margin, max_ds):
    sz_padded = 2**max_ds * np.ceil( (size_in + 2*min_margin)/2**max_ds )
    return np.max([sz_padded,[1,1]],axis=0).astype(np.int64)

def rotation_matrix_2d(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])

def gabor_2d(N, M, sigma, slant, xi, theta, no_dc=False):
    x, y = map(np.float64, np.meshgrid(np.arange(M), np.arange(N)))
    x -= np.ceil(M / 2.)
    y -= np.ceil(N / 2.)
    Rth = rotation_matrix_2d(theta)

    A = np.dot(np.linalg.solve(Rth, np.array([[1. / sigma ** 2, 0], [0, slant ** 2 / sigma ** 2]])), Rth)

    s = x * (A[0, 0] * x + A[0, 1] * y) + y * (A[1, 0] * x + A[1, 1] * y)

    gabc = np.exp(-s / 2 + 1j * (x * xi * np.cos(theta) + y * xi * np.sin(theta)))
    if no_dc:
        gabc -= (np.sum(gabc) / np.sum(np.exp(-s / 2))) * np.exp(-s / 2)

    gab = 1. / (2 * np.pi * sigma ** 2 / slant ** 2) * np.fft.fftshift(gabc)

    return gab

def periodize_filter(filter_f):
    N = np.array(filter_f.shape)

    coefft = []

    j0 = 0
    while True:
        if np.any(np.abs(np.floor(N/2.**(j0+1))-N/2.**(j0+1))>1e-6):
            break

        sz_in = [N[0]/2**j0, 2**j0, N[1]/2**j0, 2**j0]
        sz_out = N/2**j0

        filter_fj = filter_f.copy()
        for d in range(len(N)):
            mask =  np.array([1.]*int(N[d]/2.**(j0+1)) + [1./2.] + [0.]*int((1-2.**(-j0-1))*N[d]-1))
            mask += np.array([0.]*int((1-2.**(-j0-1))*N[d]) + [1./2.] + [1.]*int(N[d]/2.**(j0+1)-1))

            mask = np.atleast_2d(mask).transpose()

            if d > 0:
                mask = np.transpose(mask)

            filter_fj = filter_fj * mask


        a = np.reshape(filter_fj, sz_in, order='F')
        b = np.sum(a, 1)[:,np.newaxis,:,:]
        c = np.sum(b, 3)[:,:,:,np.newaxis]
        d = np.reshape(c, sz_out, order='F')

        coefft.append(d)

        j0 += 1

    return coefft

def realize_filter(filter_f):
    return filter_f


def morlet_filter_bank_2d(size_in, Q=1, J=4, L=8):
    size_in=np.array(size_in)
    assert size_in.shape==(2,), "size_in must be a single axis array with len=2"

    # Initialize params
    sigma_phi = 0.8
    sigma_psi = 0.8
    xi_psi = 1./2. * (2.**(-1./Q)+1) * np.pi
    slant_psi = 4./L
    min_margin = sigma_phi * 2**(float(J)/Q)

    # Calculate max resolution (downsample)
    res_max = np.floor(J/Q)

    # Calculate padded_size
    N,M = size_filter = pad_size(size_in, min_margin, res_max)

    res = 0
    # Calculate filter scale
    scale = 2.**((J-1.) / Q - res)
    # Create phi_filter and periodize it
    filter_spatial = gabor_2d(N,M, sigma_phi*scale, 1, 0, 0)
    phi_filter = np.real(np.fft.fft2(filter_spatial))
    phi_filter = periodize_filter(phi_filter)

    # Create psi_filters
    littlewood_final = np.zeros((N,M))
    angles = np.arange(L) * np.pi / L
    psi_filters = np.empty((J*L, N, M))
    meta = {
        'j': [],
        'theta': []
    }
    for j in range(J):
        for theta,angle in enumerate(angles):
            scale = 2**(float(j)/Q - res)
            filter_spatial = gabor_2d(N,M,sigma_psi*scale, slant_psi, xi_psi/scale, angle, no_dc=True)

            psi_filters[j*L+theta] = np.real(np.fft.fft2(filter_spatial))

            littlewood_final += np.abs(realize_filter(psi_filters[j*L+theta]))**2
            meta['j'].append(float(j))
            meta['theta'].append(float(theta+1))

    K = np.max(littlewood_final)
    periodized_psi_filters = []
    psi_filters /= np.sqrt(K/2.)
    for i in range(J*L):
        periodized_psi_filters.append(periodize_filter(psi_filters[i]))

    meta.update({
        'Q': Q,
        'J': J,
        'L': L,
        'sigma_phi': sigma_phi,
        'sigma_psi': sigma_psi,
        'xi_psi': xi_psi,
        'slant_psi': slant_psi,
        'size_in': size_in,
        'size_filter': size_filter
    })

    return {
        'phi': phi_filter,
        'psi': periodized_psi_filters,
        'meta': meta
    }






