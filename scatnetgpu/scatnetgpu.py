import os
import numpy as np
from scipy.special import binom

import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
import skcuda.fft as cu_fft

from filter_banks import morlet_filter_bank_2d

def get_filters(J, L, shape=None, margin=(0,0)):
    assert (shape[0]>=2**J and shape[1]>=2**J), "Shape ({}) must be less then {}".format(shape, 2**J)

    filters = morlet_filter_bank_2d(shape, J=J, L=L)

    kernels = {
        'J': J,
        'L': L,
        'j': np.array([-1]+list(filters['meta']['j'])),
        'l': np.array([0]+list(filters['meta']['theta'])),
        # 'filter' is a list of 3d-arrays. Each array is the bundle of FT of filters at different sizes.
        # First axis is the filter index, 2nd and 3rd axis are width and height in Fourier space
        'filter': [np.array([filters['phi'][n]] + [filters['psi'][i][n] for i in range(len(filters['psi']))], dtype=np.float32) for n in range(len(filters['phi']))],
        'fft': True,
        'margin': margin
    }
    kernels['filter_shape'] = map(lambda x: x.shape, kernels['filter'])
    kernels['len'] = kernels['filter_shape'][0][0]

    return kernels

def get_cache_filters(J,L,shape):
    shape = shape[:2]
    filters_cache_basepath = "filters_cache"
    if not os.path.exists(filters_cache_basepath):
        print "Filters cache dir not found, making one at: {}".format(filters_cache_basepath)
        os.mkdir(filters_cache_basepath)

    filter_path = os.path.join(filters_cache_basepath, "filter_J{}_L{}_shape{}.npz".format(J,L,shape))
    if not os.path.isfile(filter_path):
        print "Filter J:{} L:{} shape:{} not found in cache. Building it.".format(J,L,shape)
        filters = get_filters(J,L,shape)
        np.savez_compressed(filter_path, filters=filters)
    else:        
        filters = np.load(filter_path)['filters'].reshape(-1)[0]

    return filters



class SameShapeSignalContainer:
    """Allocate a batch of B images of same shape. Also stores history of j and l, and if images are ready to be processed"""

    pad_kernel = ElementwiseKernel(
        "float *dest, int h, int w, int pt, int pb, int pl, int pr",
        """
        unsigned int dest_w = w+pl+pr;
        unsigned int dest_h = h+pt+pb;
        unsigned int dest_size = dest_w*dest_h;
        
        unsigned int z = i/dest_size;          // image index
        unsigned int y = (i%dest_size)/dest_w; // padded row
        unsigned int x = (i%dest_size)%dest_w; // padded col
                
        int A = pt-y-1;
        unsigned int l = min(max(A, -A-1), 2*h+A); // original row to copy

        int B = pl-x-1;
        unsigned int k = min(max(B, -B-1), 2*w+B); // original col to copy
        
        l+=pt; // padded row to copy
        k+=pl; // padded col to copy

        dest[i] = dest[z*dest_size + l*dest_w + k];
        """,
        "padding"
    )

    def __init__(self, B, shape, original_shape, dtype=np.float32):
        try:
            shape = tuple(shape)
        except TypeError as e:
            print "Shape must be a tuple or castable to tuple, not a {}".format(type(shape))
            raise e

        self.gpu_data = gpuarray.empty((B,)+shape, dtype=dtype)
        self.j_hist = [[] for _ in xrange(B)]
        self.l_hist = [[] for _ in xrange(B)]
        self.ds_hist = [[] for _ in xrange(B)]
        self.original_shape = original_shape
        self.B = B

        padding = map(lambda a,b: (a-b)/2., self.gpu_data.shape[1:], self.original_shape)
        padding = np.int32(map(lambda x: (np.ceil(x),np.floor(x)), padding))
        self.padding = (padding[0][0], padding[0][1], padding[1][0], padding[1][1])

        self.cursor = 0 # Cursor to keep track of first empty space

    def reset_cursor(self):
        self.cursor = 0

    def set(self, imgs, full=False):
        assert len(imgs) == self.B, "Batch size must be {}, not {}".format(self.B, len(imgs))
        
        if full:
            self.gpu_data.set(imgs)
        else:
            assert imgs.shape[1:] == self.original_shape, "Image shape must be {}, not {}".format(self.original_shape, imgs.shape[1:])
            pt, pb, pl, pr = self.padding
            h, w = self.original_shape

            imgs = imgs.astype(self.gpu_data.dtype)

            for z, img in enumerate(imgs):
                self.gpu_data[z, pt:pt+h, pl:pl+w ].set(img)
                self.cursor = z+1

    def get(self, full=False):
        if full:
            return self.gpu_data.get()
        else:
            pt, pb, pl, pr = self.padding
            h, w = self.original_shape
            return self.gpu_data.get()[:,pt:pt+h, pl:pl+w]
            #return np.array([self.gpu_data[z, pt:pt+h, pl:pl+w ].get() for z in range(self.B)])

    def pad(self):
        pt, pb, pl, pr = self.padding

        h, w = self.original_shape
        self.pad_kernel(self.gpu_data, h, w, pt, pb, pl, pr)

    def get_filters(self, filters):
        total_ds = map(sum, self.ds_hist)
        assert np.min(total_ds) == np.max(total_ds), "Something is wrong here..."+total_ds

        _, filters_index = find_filter_shape(int(total_ds[0]), filters)

        j_hist = self.j_hist[0]
        if len(j_hist) == 0:
            j_hist = [-1]

        good_j_indexes = np.where( (filters['j']>j_hist[-1])  + (filters['j']==-1))[0]
        filters_gpu = gpuarray.to_gpu(filters['filter'][filters_index][good_j_indexes])
        filters_j = filters['j'][good_j_indexes]
        filters_l = filters['l'][good_j_indexes]

        return filters_gpu, filters_j, filters_l

    def get_image_offset(self, n):
        img_size = self.gpu_data.shape[1]*self.gpu_data.shape[2]
        return n*img_size


def get_signals_count(M,J,L):
    """Return a list of number of images for each downsample factor (shape/2**ds) that will appear in a scattering network with given M, J, L. List index = downsample factor from original shape"""
    counts = []
    s = 0
    for j in range(1,J+1):
        x = sum([L**q * int(binom(j, q)) for q in range(M+1)][1:]) - s
        s += x
        counts.append(x)
    counts.append(s+1)

    return counts

def allocate_containers(M, img_shape, filters):
    J = filters['J']
    L = filters['L']
    Bs = get_signals_count(M,J,L)

    # Create a list of container
    containers = []

    # Add container for original image
    padded_shape, filter_index = find_filter_shape(0, filters)
    c = SameShapeSignalContainer(1, padded_shape, img_shape)
    containers.append(c)

    # Add containers for all intermediate signals
    for ds, B in enumerate(Bs):
        ds_shape = map(lambda s: int(np.ceil(s/2.**ds)), img_shape)
        padded_shape, filter_index = find_filter_shape(ds, filters)
        c = SameShapeSignalContainer(B, padded_shape, ds_shape)
        containers.append(c)

    return containers

def find_filter_shape(resolution, filters):
    len_filters = len(filters['filter_shape'])
    assert 0 <= resolution <= len_filters, "Wrong resolution. Filters len is {} and you requested resolution {}".format(len_filters, resolution)
    if resolution == len_filters:
        return map(lambda x: int(np.ceil(x/2.)), filters['filter_shape'][resolution-1][1:]), resolution
    else:
        return filters['filter_shape'][resolution][1:], resolution


filter_multiply = ElementwiseKernel(
    """
    float                  *f,       
    unsigned int            hw,      
    pycuda::complex<float> *s,       
    unsigned int           *s_id,    
    unsigned int            len_s_id,
    pycuda::complex<float> *out,     
    unsigned int            offset
    """,
    """
    /* Arguments
     * *f,        // filters vector
     *  hw,       // filter size (h*w)
     * *s,        // signals vector
     * *s_id,     // vector of indices of signals to multiply
     *  len_s_id, // len of s_id
     * *out,      // output vector
     *  offset    // offset of output vector index
     */

    /* Indices:
     * i -> *f
     * j -> *s
     * k -> *out
     * a -> *s_id
     */

    unsigned int xy = i%(hw);

    float fi = f[i];
    
    unsigned int j, k;

    for (int a=0; a<len_s_id; ++a) {

        j = s_id[a]*hw + xy;

        k = offset + a*hw + xy;

        out[k] = s[j] * fi;
        
    }
    """,
    "filter_multiply"
)

unpad_ds_modulo = ElementwiseKernel(
    """
    pycuda::complex<float> *src,
    float                  *dest,
    unsigned int           *signals_idx,

    unsigned int           stride_y,
    unsigned int           stride_x,

    unsigned int           unpadded_src_h,
    unsigned int           unpadded_src_w,
    unsigned int           src_pt,
    unsigned int           src_pb,
    unsigned int           src_pl,
    unsigned int           src_pr,

    unsigned int           unpadded_dest_h,
    unsigned int           unpadded_dest_w,
    unsigned int           dest_pt,
    unsigned int           dest_pb,
    unsigned int           dest_pl,
    unsigned int           dest_pr,

    unsigned int           dest_offset
    """,
    """
    /* Indices and coordinates:
     * i: index in unpadded dest batch
     * j: index in padded dest batch
     * k: index in padded src batch
     * *_x: col
     * *_y: row
     * *_z: image index in batch
     */

    unsigned int unpadded_dest_hw = unpadded_dest_h * unpadded_dest_w;
    unsigned int unpadded_dest_z  = (i / unpadded_dest_hw);
    unsigned int unpadded_dest_y  = (i % unpadded_dest_hw) / unpadded_dest_w;
    unsigned int unpadded_dest_x  = (i % unpadded_dest_hw) % unpadded_dest_w;


    unsigned int padded_dest_h    = unpadded_dest_h + dest_pt + dest_pb;
    unsigned int padded_dest_w    = unpadded_dest_w + dest_pl + dest_pr;
    unsigned int padded_dest_hw   = padded_dest_h * padded_dest_w;
    unsigned int padded_dest_z    = unpadded_dest_z;
    unsigned int padded_dest_y    = unpadded_dest_y + dest_pt;
    unsigned int padded_dest_x    = unpadded_dest_x + dest_pl;

    unsigned int j = dest_offset + padded_dest_z * padded_dest_hw + padded_dest_y * padded_dest_w + padded_dest_x;

    
    unsigned int padded_src_h     = unpadded_src_h + src_pt + src_pb;
    unsigned int padded_src_w     = unpadded_src_w + src_pl + src_pr;
    unsigned int padded_src_hw    = padded_src_h * padded_src_w;
    unsigned int padded_src_z     = signals_idx[padded_dest_z];
    unsigned int padded_src_y     = src_pt + unpadded_dest_y * stride_y;
    unsigned int padded_src_x     = src_pl + unpadded_dest_x * stride_x;

    unsigned int k = padded_src_z * padded_src_hw + padded_src_y * padded_src_w + padded_src_x;


    dest[j] = pycuda::abs(src[k]) * stride_x;
    """,
    "unpad_ds_modulo"
)

def fft_gpu(img_gpu, img_f_gpu=None, stream=None):
    """Execute FFT on the GPU. img is a np.array of reals. It returns a GPUArray containing the complex Fourier transform of img"""
    assert np.prod(img_gpu.shape)>0
    assert img_gpu.dtype == np.complex64

    if img_f_gpu is None:
        # Allocate memory on GPU for the transform of img
        img_f_gpu = gpuarray.empty(img_gpu.shape, np.complex64)

    # Prepare plan
    batch = (img_gpu.shape[0] if len(img_gpu.shape)==3 else 1)
    plan_forward = cu_fft.Plan(img_gpu.shape[-2:], np.complex64, np.complex64, stream=stream, batch=batch)

    # Execute FFT
    cu_fft.fft(img_gpu.astype(np.complex64), img_f_gpu, plan_forward, False) # Not sure why, but this must be False

    # Return the GPU array
    return img_f_gpu

def ifft_gpu(img_f_gpu, img_gpu=None, stream=None):
    '''Execute Inverse FFT on the GPU. img_f_gpu is a GPUArray. Returns a GPUArray containing inverse transform of img_f_gpu'''
    assert np.prod(img_f_gpu.shape) > 0
    assert img_f_gpu.dtype == np.complex64

    if img_gpu is None:
        # Allocate memory for the inverse transform
        print "Allocating outupt gpuarray"
        img_gpu = gpuarray.empty(img_f_gpu.shape, np.complex64)
    
    # Prepare plan
    batch = (img_f_gpu.shape[0] if len(img_f_gpu.shape)==3 else 1)
    plan_backward = cu_fft.Plan(img_f_gpu.shape[-2:], np.complex64, np.complex64, stream=stream, batch=batch)

    # Execute IFFT
    cu_fft.ifft(img_f_gpu, img_gpu, plan_backward, True) # Not sure why, but this must be True

    # Return the GPU array
    return img_gpu



def scattering_newtork(img, filters, containers=None, M=2):

    if containers is None:
        containers = allocate_containers(M, img.shape, filters)
    else:
        for c in containers:
            c.reset_cursor()


    # Move src image to gpu in correct container
    containers[0].set(np.expand_dims(img, axis=0))

    # Containers list must be walked in order
    for c_i, c in enumerate(containers[:-1]):

        # Pad signals in container
        c.pad()

        # FFT of signals
        signals_f = fft_gpu(c.gpu_data.astype(np.complex64))

        # Find filters
        filters_f_gpu, filters_j, filters_l = c.get_filters(filters)

        # Count how many outputs there will be
        n_outs = 0
        for i, j_hist in enumerate(c.j_hist):
            if len(j_hist) < M:
                n_outs += len(filters_j)
            else:
                n_outs += np.sum(filters_j == -1)
        
        # Prepare an empty gpuarray for multiplication outputs
        data_gpu = gpuarray.empty((n_outs,) + signals_f.shape[1:], np.complex64)

        data_new_j = []
        data_new_l = []
        data_orig_id = []
        # For each filter
        for y, (j, l) in enumerate(zip(filters_j, filters_l)):

            if j == -1: # Low pass filter, convolve with all signals
                signals_idx = np.arange(len(signals_f))
            
            else: # High pass filter, convolve only with len(j_hist) < M
                j_hist_lens = np.array(map(len, c.j_hist))
                signals_idx = np.where(j_hist_lens < M)[0]


            _,h,w = filters_f_gpu.shape

            hw = h*w

            signals_idx_gpu = gpuarray.to_gpu(signals_idx.astype(np.uint32))
            filter_multiply(filters_f_gpu, hw, signals_f, signals_idx_gpu, len(signals_idx), data_gpu, len(data_orig_id)*hw, slice=slice(y*hw, (y+1)*hw, 1))

            data_new_j += [j] * len(signals_idx)
            data_new_l += [l] * len(signals_idx)
            data_orig_id += signals_idx.tolist()


        # Perform IFFT
        ifft_gpu(data_gpu, data_gpu)


        # Prepare to unpad, downsample and modulo
        # List of num_containers empty lists that will contain id of signals that needs to be stored in each container
        signals_to_containers = [[] for i in range(len(containers))] 

        for k, (i, j, l) in enumerate(zip(data_orig_id, data_new_j, data_new_l)):
            total_ds = sum(c.ds_hist[i])

            if j==-1:
                ds = int(max(0, filters['J'] - total_ds))
            else:
                ds = int(max(0, j-total_ds))

            new_total_ds = total_ds+ds

            # index is plus 1 becouse first container is source image
            dest_container_idx = new_total_ds+1
            signals_to_containers[dest_container_idx].append(k)

            dest_container = containers[dest_container_idx]
            dest_container.j_hist[dest_container.cursor]  = c.j_hist[i]  + [int(j)]
            dest_container.l_hist[dest_container.cursor]  = c.l_hist[i]  + [int(l)]
            dest_container.ds_hist[dest_container.cursor] = c.ds_hist[i] + [ds]
            dest_container.cursor += 1


        for c_id, signals_idx in enumerate(signals_to_containers):
            signals_idx = np.array(signals_idx, dtype=np.uint32)
            if len(signals_idx) > 0:

                # Load dest container
                dest_container = containers[c_id]

                # Index i in elwisekernel will span from 0 to size of unpadded image batch. It will be used to map from src to dest arrays.
                kernel_range = slice(0, len(signals_idx)*dest_container.original_shape[0]*dest_container.original_shape[1], 1)

                # Stride to downsample
                stride_y = stride_x = 2**dest_container.ds_hist[dest_container.cursor-1][-1]
                assert stride_x==stride_y

                # Src padding
                s_pt, s_pb, s_pl, s_pr = c.padding

                # Src original h and w
                s_h, s_w = c.original_shape

                # Dest padding
                d_pt, d_pb, d_pl, d_pr = dest_container.padding

                # Dest original h and w
                d_h, d_w = dest_container.original_shape

                # Dest base offset
                d_offset = dest_container.get_image_offset(dest_container.cursor - len(signals_idx))

                # Signals to map to this dest container
                # Try if it work without manual transfer
                signals_idx_gpu = gpuarray.to_gpu(signals_idx)

                # Launch kernel
                unpad_ds_modulo(data_gpu, dest_container.gpu_data, signals_idx_gpu, stride_y, stride_x, s_h, s_w, s_pt, s_pb, s_pl, s_pr, d_h, d_w, d_pt, d_pb, d_pl, d_pr, d_offset, range=kernel_range)


    signals = containers[-1].get()

    j_hist_list = containers[-1].j_hist
    l_hist_list = containers[-1].l_hist
    ds_hist_list = containers[-1].ds_hist


    S = []
    for m in range(M+1):
        S_tmp = {'j': [], 'l': [], 'ds': [], 'signal': []}

        signals_idx = [i for i in range(len(j_hist_list)) if len(j_hist_list[i]) == 1+m]

        # Sort like original code
        argosrt = np.argsort(['x'.join(["{:09d}x{:09d}".format(j,l) for j,l in zip(*jl_hist)]) for (idx, jl_hist) in enumerate(zip(j_hist_list, l_hist_list)) if len(jl_hist[0]) == 1+m], axis=0)

        for i in argosrt:
            i = signals_idx[i]
            S_tmp['j'].append(j_hist_list[i])
            S_tmp['l'].append(l_hist_list[i])
            S_tmp['ds'].append(ds_hist_list[i])
            S_tmp['signal'].append(signals[i])

        S.append(S_tmp)

    return S, None #, containers


def stack_scat_output(S):
        try:
            _ = S[0].dtype
            # If not fail, it's octave output
            output = []
            for s in S:
                for im in s['signal'][0][0]:
                    output.append(im)
            return np.array(output)
        except:
            # Else, it's pycuda output
            output = []
            paths = []
            for s in S:
                for im,j,l in zip(s['signal'],s['j'],s['l']):
                    output.append(im)
                    j = j[:-1] if len(j) > 1 else j
                    l = l[:-1] if len(l) > 1 else l

                    paths.append((j,l))
            return np.array(output), paths


class ScatNet:
    def __init__(self, M, J, L, shape=None):
        self.M = M
        self.J = J
        self.L = L
        self.shape = None

        self.filters = None
        self.containers = None

        if shape is not None:
            self.prepare_for_shape(shape)

    def prepare_for_shape(self, shape):
        if not self.shape == shape[:2]:
            self.shape = shape[:2]
            self.filters = get_cache_filters(self.J, self.L, self.shape)

            self.containers = allocate_containers(self.M, self.shape, self.filters)

    def transform(self, img):
        self.prepare_for_shape(img.shape)

        ''' Trasform img with the scatterin network. If shape is 3 it's assumed that last axis are channels and each channel is scattered independetly'''
        if len(img.shape) == 2:
            S, _ = scattering_newtork(img, self.filters, containers=self.containers, M=self.M)
            return S
        elif len(img.shape) == 3:
            c_outs = []
            for c in range(img.shape[2]):
                S = self.transform(img[:,:,c])
                c_outs.append(S)
            return c_outs

    def transform_batch(self, imgs):
        out = []

        for img in imgs:
            S = self.transform(img)
            out.append(S)
        return out