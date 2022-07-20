"""Create lmdb dataset"""
from numpy.core.fromnumeric import reshape
from util import *
import lmdb
import caffe


def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2020):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        # import ipdb; ipdb.set_trace()
        new_data = []
        data = minmax_normalize(data)
        # data = minmax_normalize(data.transpose((2,0,1)))
        # Visualize3D(data)
        # import ipdb; ipdb.set_trace()
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])
        for i in range(len(scales)):
            temp = zoom(data, zoom=(1, scales[i], scales[i]))
            # import ipdb; ipdb.set_trace()
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    print(data.shape)
    data = preprocess(data)
    N = data.shape[0]
    
    print(data.shape)

    # We need to prepare the database for the size. We'll set it 2 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
#    import ipdb; ipdb.set_trace()
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    # env = lmdb.open(name+'.db', map_size=int(1e11), writemap=True)
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            # import ipdb; ipdb.set_trace()
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)        
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')


def create_lmdb_test(
    datadir, fns, name, matkey,
    preprocess, 
    load=h5py.File, 
    seed=2020):
    """
    Create Augmented Dataset
    """
    np.random.seed(seed)

    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]
    
    print(data.shape)

    # We need to prepare the database for the size. We'll set it 2 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 1.25
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
#    import ipdb; ipdb.set_trace()
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i, fn in enumerate(fns):
            X = preprocess(load(datadir + fn)[matkey])
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(i*N+j)
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')


# def create_icvl64_31():
#     print('create icvl64_31...')
#     datadir = 'Data/ICVL/Training/'
#     # fns = os.listdir('Data/ICVL/ICVL_RGB')
#     fns = open('Data/ICVL/full_as_in_eccv_paper.txt').readlines()
#     fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
#     create_lmdb_train(
#         datadir, fns, '/home/kaixuan/Dataset/ICVL64_31', 'rad', 
#         crop_sizes=(768, 768),
#         scales=(1,),
#         ksizes=(31, 64, 64),
#         strides=(31, 32, 32),
#         load=h5py.File, augment=True,
#     )
'''
def create_Salinas():
    print('create Salinas...')
    datadir = '/media/kaixuan/DATA/Papers/Code/Matlab/ITSReg/code of ITSReg MSI denoising/data/real/new/Salinas/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/media/kaixuan/DATA/Papers/Code/Data/remote/Salinas', 'hsi', 
        crop_sizes=None,
        scales=(1,),
        ksizes=(197, 64, 64),
        strides=[(197, 32, 32)],
        load=loadmat, augment=True,
    )
'''

def create_Pavia():
    print('create Pavia...')
    datadir = '/data/liangzy/Data/remote/train/'
    fns = os.listdir(datadir)
#    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    fns = ['pavia1.mat', 'pavia2.mat']

    create_lmdb_train(
        datadir, fns, '/data/liangzy/Data/remote/train/pavia_reshape', 'hsi', 
        crop_sizes=None,
        scales=(1,),
        ksizes=(102, 64, 64),
        strides=[(102, 32, 32)],
        load=loadmat, augment=True,
    )

def create_PaviaU():
    print('create PaviaU...')
    datadir = '/data/liangzy/Data/remote/train/'
    fns = os.listdir(datadir)
#    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    fns = ['paviaU1.mat', 'paviaU2.mat']

    create_lmdb_train(
        datadir, fns, '/data/liangzy/Data/remote/train/paviaU', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(103, 64, 64),
        strides=[(103, 32, 32)],
        load=loadmat, augment=True,
    )

def create_Salinas():
    print('create Pavia...')
    datadir = '/data/liangzy/Data/remote/train/'
    fns = os.listdir(datadir)
#    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    fns = ['salinas1.mat', 'salinas2.mat']

    create_lmdb_train(
        datadir, fns, '/data/liangzy/Data/remote/train/salinas', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(224, 64, 64),
        strides=[(224, 32, 32)],
        load=loadmat, augment=True,
    )

def create_Urban():
    print('create Urban...')
    datadir = '/media/liangzy/Data/Data/urban-v1/train/'
    fns = os.listdir(datadir)
#    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    fns = ['urban1.mat', 'urban2.mat', 'urban3.mat', 'urban4.mat']

    create_lmdb_train(
        datadir, fns, '/media/liangzy/Data/Data/urban-v1/train/urban', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(162, 64, 64),
        strides=[(162, 32, 32)],
        load=loadmat, augment=True,
    )

def create_icvl192_31():
    print('create icvl192_31...')
    datadir = '/data/weikaixuan/hsi/data/Training/'
    fns = os.listdir(datadir)
    # fns = open('Data/ICVL/full_as_in_eccv_paper.txt').readlines()
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/data/liangzy/Data/icvl_train/ICVL192_31', 'rad', 
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 192, 192),
        strides=[(31, 192, 192), (31, 96, 96), (31, 96, 96)],
        load=h5py.File, augment=True,
    )

def create_icvl256_31():
    print('create icvl256_31...')
    datadir = '/BIT/BIT/liangzhiyuan/icvl/icvl100/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/BIT/BIT/liangzhiyuan/ICVL256_31_100', 'rad',
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 256, 256),
        strides=[(31, 256, 256), (31, 128, 128), (31, 128, 128)],
        load=h5py.File, augment=True,
    )


def create_icvl64_31():
    print('create icvl64_31...')
    # datadir = '/media/liangzy/Data/Data/icvl/small/train/'
    datadir = '/media/exthdd/datasets/hsi/icvl/icvl100/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/media/exthdd/datasets/hsi/lzy_data/ICVL64_31_100_20210621', 'rad',
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        load=h5py.File, augment=True,
    )


def create_icvl64_31_small():
    print('create icvl64_31...')
    # datadir = '/media/liangzy/Data/Data/icvl/small/train/'
    datadir = '/media/exthdd/datasets/hsi/icvl/icvl100/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    fns = fns[0:30]

    create_lmdb_train(
        datadir, fns, '/media/exthdd/datasets/hsi/lzy_data/small/ICVL64_31_30', 'rad',
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        load=h5py.File, augment=True,
    )


def create_cave_512_31():
    def preprocess(data):
        new_data = minmax_normalize(data)
        new_data = new_data.transpose((2,0,1))
        return new_data[None]

    print('create cave512_31...')
    datadir = 'Data/CAVE/Training/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_test(
        datadir, fns, 
        '/home/kaixuan/Dataset/CAVE512_31', 
        'truth', 
        preprocess=preprocess,
        load=loadmat
    )

def create_icvl32_31():
    print('create icvl32_31...')
    datadir = '/data/weikaixuan/hsi/data/Training/'
    fns = os.listdir(datadir)
    # fns = open('Data/ICVL/full_as_in_eccv_paper.txt').readlines()
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/data/liangzy/Data/icvl_train/ICVL32_31', 'rad',
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25, 0.125),
        ksizes=(31, 32, 32),
        strides=[(31, 32, 32), (31, 16, 16), (31, 16, 16), (31, 32, 32)],
        load=h5py.File, augment=True,
    )


# For data_add and data_sub
def create_stats_train(datadir, fns, matkey, load=h5py.File):
    mean = 0
    std = 0
    for i, fn in enumerate(fns):
        try:
            X = load(datadir + fn)[matkey]
        except:
            print('loading', datadir+fn, 'fail')
            continue
        X = minmax_normalize(X)
        X = reshape(X, (X.shape[0], X.shape[-1]*X.shape[-2]))
        print(X.shape)
        mean += np.mean(X, axis=1)
        std += np.std(X, axis=1)
    print('done')
    mean = mean.astype(np.float32) / len(fns)
    std = std.astype(np.float32) / len(fns)
    np.savez('icvl_stats.npz', data_mean=mean, data_std=std, dtype='float32')



def create_icvl64_31_stats():
    print('create icvl means and stds...')
    # datadir = '/media/liangzy/Data/Data/icvl/small/train/'
    datadir = '/media/exthdd/datasets/hsi/icvl/icvl100/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    # fns = ['nachal_0823-1040.mat']

    create_stats_train(
        datadir, fns, 'rad', load=h5py.File
    )


def create_harvard64_31():
    print('create harvard64_31...')
    datadir = '/media/exthdd/datasets/hsi/harvard_mat/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/media/exthdd/datasets/hsi/lzy_data/Harvard64_31_55', 'ref',
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        load=h5py.File, augment=True,
    )


def create_cave64_31():
    print('create harvard64_31...')
    datadir = '/mnt/e/Data/CAVE/cave_512_15/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns[0:24]]

    create_lmdb_train(
        datadir, fns, '/mnt/e/Data/CAVE/CAVE64_31_24', 'gt',
        crop_sizes=None,
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],
        load=h5py.File, augment=True,
    )


if __name__ == '__main__':
    create_harvard64_31()
    # create_icvl64_31_small()
    # create_icvl64_31_stats()
    pass

