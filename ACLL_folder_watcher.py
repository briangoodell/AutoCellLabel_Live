import importlib
import os
import nrrd
import nd2
import numpy as np
import h5py

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from torch.profiler import profile, record_function, ProfilerActivity

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = utils.get_logger('UNet3DPredict')


def _get_output_file(dataset, suffix='_predictions'):
    loc_path = "../predictions/"
    head, tail = os.path.split(dataset.file_path)
    if not os.path.exists(os.path.join(head, loc_path)):
        os.mkdir(os.path.join(head, loc_path))
    name = os.path.splitext(tail)[0]
    name = name.split("_")[0]  # remove frame number
    return f'{os.path.join(head, loc_path)}{name}{suffix}.h5'


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device).eval()
    # model = model.to(memory_format=torch.channels_last_3d)

    folder = "F:\\Brian\\Live"
    # folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw"
    folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw_test"
    # folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/H5_raw_full_dataset/train"
    # folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/train"
    # warmup_file = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw/2022-07-27-31_t0001_ch2.nrrd"
    assert os.path.exists(folder), "Folder does not exist"
    # assert not os.path.exists(os.path.join(folder, "liveTraces.h5")), "liveTraces.h5 already exists in the folder"

    predictor = _get_predictor(model, None, os.path.join(folder, "liveTraces_mergeshear.h5"), config) # We can have just one predictor, because we are saving it all to one file

    ran = []

    ##### WARMUP #####
    # data, header = nrrd.read(warmup_file)
    # red = np.ascontiguousarray(data[...])
    # static_red = torch.empty(red.shape, device = device)          # will be overwritten each call
    # static_green = torch.empty(red.shape, device = device)          # will be overwritten each call
    # static_y = None
    # with torch.inference_mode():
    #     for _ in range(5):
    #         _ = predictor.predict(static_red, static_green)

    # g = torch.cuda.CUDAGraph()
    # torch.cuda.synchronize()
    # with torch.inference_mode(), torch.cuda.graph(g):
    #     static_y = predictor.predict(static_red, static_green)

####### TODO: Warmup & inform user when ready.

    # def my_handler(prof): 
    #         print("Trace ready") 
    #         torch.profiler.tensorboard_trace_handler("/home/brian/data4/brian/freelyMoving/profiling")(prof) 
            
    # with torch.profiler.profile( 
    #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1), 
    #     on_trace_ready=my_handler, record_shapes=True, profile_memory=True, with_stack=True, 
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] ) as prof:

    x_tot = 0.0
    y_tot = 0.0
    th_tot = 0.0
    chan_calb_count = 0
    calc_chan_align_params = None


    while True:
        files = [x for x in os.listdir(folder) if x.endswith(".nd2") or x.endswith("ch2.nrrd") or (x.endswith(".h5") and "liveTraces" not in x)]
        # files = [x for x in os.listdir(folder) if x.endswith(f"_{len(ran) + 1}.h5")]

        # files = [x for x in files if "2022-07-27-31" in x]
        
        # for file in sorted(files, key=lambda x: int(x.split('_')[-1].split(".")[0])):
        for file in sorted(files, key=lambda x: int(x.split('_t')[-1].split("_ch")[0])):
        # for file in files:
            if file not in ran:
                # if file.endswith(".nd2"):
                #     print("Processing " + file)
                #     #data, header = nrrd.read(os.path.join(folder, file))
                #     data = nd2.imread(os.path.join(folder, file))

                #     print(data.shape)
                #     # red = np.expand_dims(data[8:72, 1, 135:495, 57:909], 0)
                #     # green = np.expand_dims(data[8:72, 0, 135:495, 57:909], 0)
                #     red = data[8:72, 1, 135:495, 57:909]
                #     green = data[8:72, 0, 135:495, 57:909]
                #     H, W = red.shape[-2:]
                #     red = red.reshape(*red.shape[:-2], H // 3, 3, W // 3, 3)
                #     red = red.sum(axis=(-1, -3))
                #     green = green.reshape(*green.shape[:-2], H // 3, 3, W // 3, 3)
                #     green = green.sum(axis=(-1, -3))
                #     print(red.shape)

                #elif
                if file.endswith("ch2.nrrd"):
                    print("Processing NRRD (assumed binned) " + file)
                    data, header = nrrd.read(os.path.join(folder, file))
                    # print(data.shape)
                    data = data.transpose(2, 1, 0)  # XYZ to ZYX
                    # red = np.expand_dims(data[...], 0)
                    red = data[8:72, 45:165, 19:303] # Naive cropping
                    # red = data[8:72, 135:495, 57:909] # Naive cropping
                    red = np.ascontiguousarray(red)
                    # red = data
                    data, header = nrrd.read(os.path.join(folder, file.replace("ch2", "ch1")))
                    data = data.transpose(2, 1, 0)  # XYZ to ZYX
                    # green = data[...]
                    green = data[8:72, 45:165, 19:303] # Naive cropping
                    green = np.ascontiguousarray(green)

                    # r = red.to(device, non_blocking=True)
                    # static_red.copy_(r, non_blocking=True)
                    # g = red.to(device, non_blocking=True)
                    # static_green.copy_(g, non_blocking=True)

                    
                    t = int(file.split("t")[-1].split("_")[0])
                    # green = data

                elif file.endswith(".h5"):
                    if "2022-07-27-31" not in file:
                        continue
                    
                    
                    with h5py.File(os.path.join(folder, file), 'r') as f:
                        red = np.ascontiguousarray(f["raw"][0])
                        green = red

                        # green = np.ascontiguousarray(f["raw_green"][0])
                        # labels = torch.tensor(f["label"][:], device=device)

                    print("Processing h5 " + file)
                    t = int(file.split("_")[-1].split(".")[0])
                    

                
                else:
                    continue
                
                D,H,W = red.shape
                print(f"Array size: D:{D} H:{H} W:{W}")

                # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/H5_raw_full_dataset/train/2022-07-27-31_{t}.h5", "r") as f:
                #     labels = torch.tensor(f["label"][:], device=device)

                # I'm just going to say from now on, [D, H, W] is the order we're going with
                chan_align_params = predictor.predict(red, green, chan_align_params=calc_chan_align_params)
                # chan_align_params = predictor.predict(red, green, frame=t)
                # chan_align_params = predictor.predict(red, green, labels = labels)
                x_tot += chan_align_params[0]
                y_tot += chan_align_params[1]
                th_tot += chan_align_params[2]
                chan_calb_count += 1
                if chan_calb_count == 10:
                    calc_chan_align_params = (x_tot/chan_calb_count,
                                                y_tot/chan_calb_count,
                                                th_tot/chan_calb_count)
                # print(chan_align_params)

                # g.replay()
                ran.append(file)
                # prof.step()
            




if __name__ == '__main__':
    main()

    
