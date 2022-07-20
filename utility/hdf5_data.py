"""create hdf5 datasets
Warning: using hdf5 dataset in pytorch may cause several problems (thread-safe etc..)
"""
from util import *


def create_dataset(datadir, nchw, name, matkey, train_ratio=0.9, load=loadmat, preprocess=None, seed=2017):
    if not preprocess:
        preprocess = lambda identity: identity
    
    with h5py.File(name+'.h5', 'a') as f:
        np.random.seed(seed)
        fns = np.random.permutation([fn for fn in os.listdir(datadir) if fn.endswith('.mat')])        
        # Train-val split 
        part_idx = int(np.floor(train_ratio * len(fns)))
        train_fns, val_fns = fns[:part_idx], fns[part_idx:]
        
        n, c, h, w = nchw
        train = f.create_dataset(
            'Train', 
            dtype='f',
            shape=(n*len(train_fns), c, h, w),
            chunks = (1,c,h,w),
            maxshape=(None, c, h, w)
            )

        val = f.create_dataset(
            'Val', 
            dtype='f', 
            shape=(n*len(val_fns), c, h, w),
            chunks = (1,c,h,w),
            maxshape=(None, c, h, w)
            )
        
        train.attrs.create('names', train_fns, dtype=h5py.special_dtype(vlen=str))
        val.attrs.create('names', val_fns, dtype=h5py.special_dtype(vlen=str))
        
        for i, fn in enumerate(train_fns):
            mat = load(datadir + fn)
            train[i*n:(i+1)*n,:,:,:] = preprocess(mat[matkey])
            print('(train) load mat (%d/%d): %s' %(i,len(train_fns),fn))

        for i, fn in enumerate(val_fns):
            mat = load(datadir + fn)
            val[i*n:(i+1)*n,:,:,:] = preprocess(mat[matkey])
            print('(val) load mat (%d/%d): %s' %(i,len(val_fns),fn))

        print('done')


def create_dataset_v2(datadir, nchw, name, matkey, load=loadmat, preprocess=None, seed=2017):
    """
    Create Dataset without train-val split
    """
    if not preprocess:
        preprocess = lambda identity: identity
    
    with h5py.File(name+'.h5', 'a', libver='latest') as f:
        f.swmr_mode = True

        np.random.seed(seed)
        train_fns = np.random.permutation([fn for fn in os.listdir(datadir) if fn.endswith('.mat')])
        
        n, c, h, w = nchw
        train = f.create_dataset(
            'Data',
            dtype='f',
            shape=(n*len(train_fns), c, h, w),
            chunks = (1,c,h,w),
            maxshape=(None, c, h, w)
            )
        
        for i, fn in enumerate(train_fns):
            mat = load(datadir + fn)
            
            train[i*n:(i+1)*n,:,:,:] = preprocess(mat[matkey])
            print('(train) load mat (%d/%d): %s' %(i,len(train_fns),fn))

        print('done')


def create_dataset_v3(
    datadir, fns, name, matkey, 
    crop_size, scales, ksizes, strides,
    n=None, load=h5py.File, augment=True,
    seed=2017, shuffle=True):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        data = data[:,-crop_size:, -crop_size:]
        for i in range(len(scales)):
            temp = zoom(data, zoom=(1, scales[i], scales[i]))
            temp = Data2Volume(temp, ksizes=ksizes, strides=strides)
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data

    np.random.seed(seed)

    # calculate the shape of dataset
    if n is None:
        data = load(datadir + fns[0])[matkey]
        data = preprocess(data)
        n = data.shape[0]
        
    c, h, w = ksizes

    with h5py.File(name+'.h5', 'a', libver='latest') as f:
        f.swmr_mode = True
                
        train = f.create_dataset(
            'Data',
            dtype='f',
            shape=(n*len(fns), c, h, w),
            chunks = (1,c,h,w),
            maxshape=(None, c, h, w)
            )
        
        for i, fn in enumerate(fns):
            mat = load(datadir + fn)
            train[i*n:(i+1)*n,:,:,:] = preprocess(mat[matkey])
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        if shuffle:
            print('shuffle dataset...')
            np.random.shuffle(train)
        print('done')


def create_cave():
    print('create cave...')
    datadir = 'CAVE/mat_data/Training/'
    nchw = 1, 31, 512, 512
    create_dataset(datadir, nchw, 'CAVE','truth', preprocess=None)


def create_cave64():
    print('create cave64...')
    datadir = 'CAVE/Temp64/'
    nchw = 289, 31, 64, 64
    create_dataset(datadir, nchw, 'CAVE64','Volume', preprocess=minmax_normalize)


def create_icvl64():
    print('create icvl64...')
    datadir = 'ICVL/Training/'
    nchw = 289, 31, 64, 64
    hsi_rot = partial(np.rot90, k=2, axes=(1,2))    
    crop = lambda img: img[:,-1024:, -1024:]
    zoom_512 = partial(zoom, zoom=[1, 0.51, 0.51])
    d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    preprocess = sequetial_process(hsi_rot, crop, zoom_512, minmax_normalize, d2v)
    create_dataset(datadir, nchw, 'ICVL64', 'rad', load=h5py.File, preprocess=preprocess)


def create_icvl64_v2():
    print('create icvl64_v2...')
    datadir = 'ICVL/Training/'
    nchw = 289, 31, 64, 64
    hsi_rot = partial(np.rot90, k=2, axes=(1,2))
    crop = lambda img: img[:,-1024:, -1024:]
    zoom_512 = partial(zoom, zoom=[1, 0.51, 0.51])
    d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    preprocess = sequetial_process(hsi_rot, crop, zoom_512, minmax_normalize, d2v)
    create_dataset_v2(datadir, nchw, 'ICVL64', 'rad', load=h5py.File, preprocess=preprocess)


def create_icvl64_v2():
    print('create icvl64_v2...')
    datadir = 'Data/ICVL/Training/'
    nchw = 289, 31, 64, 64
    hsi_rot = partial(np.rot90, k=2, axes=(1,2))
    crop = lambda img: img[:,-1024:, -1024:]
    zoom_512 = partial(zoom, zoom=[1, 0.51, 0.51])
    d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    preprocess = sequetial_process(hsi_rot, crop, zoom_512, minmax_normalize, d2v)
    create_dataset_v2(datadir, nchw, 'ICVL64', 'rad', load=h5py.File, preprocess=preprocess)


def create_icvl64_v2():
    print('create icvl64_v2...')
    datadir = 'Data/ICVL/Training/'
    nchw = 289, 31, 64, 64
    hsi_rot = partial(np.rot90, k=2, axes=(1,2))
    crop = lambda img: img[:,-1024:, -1024:]
    zoom_512 = partial(zoom, zoom=[1, 0.51, 0.51])
    d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    preprocess = sequetial_process(hsi_rot, crop, zoom_512, minmax_normalize, d2v)
    create_dataset_v2(datadir, nchw, 'ICVL64', 'rad', load=h5py.File, preprocess=preprocess)
    

def create_icvl32_v3():
    print('create icvl32_v3...')
    datadir = 'Data/ICVL/Training/'
    fns = os.listdir('Data/ICVL/ICVL_RGB')
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_dataset_v3(
        datadir, fns, 'ICVL32', 'rad', 
        crop_size=1024, 
        scales=[0.5, 0.25, 0.125],
        ksizes=[31, 32, 32],
        strides=[1, 16, 16], n=1235,
        load=h5py.File, augment=True, shuffle=True
    )


def create_cave_test():
    print('create cave_test...')
    datadir = 'Data/CAVE/mat_data/Training/'
    nchw = 1, 31, 512, 512
    transpose = lambda img: img.transpose((2,0,1))
    create_dataset_v2(datadir, nchw, 'CAVE','truth', preprocess=transpose)


if __name__ == '__main__':
    # create_icvl32_v3()
    # create_cave_test()
    # create_icvl64_v2()
    pass
