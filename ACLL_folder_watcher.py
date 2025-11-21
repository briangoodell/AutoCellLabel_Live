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
    # folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/test"
    # warmup_file = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw/2022-07-27-31_t0001_ch2.nrrd"

    ## TEMP
    input_paths = {
        "prj_rim": "/store1/prj_rim/data_processed",
        "prj_neuropal": "/store1/prj_neuropal/data_processed",
        "prj_starvation": "/data1/prj_starvation/data_processed",
        "prj_5ht": "/data3/prj_5ht/published_data/data_processed_neuropal",
        "prj_aversion": "/data1/prj_aversion/data_processed"
    }
    datasets_prj_neuropal = ["2022-07-15-06", "2022-07-15-12", "2022-07-20-01", "2022-07-26-01", "2022-08-02-01", "2023-01-23-08", "2023-01-23-15", "2023-01-23-21", "2023-01-19-08", "2023-01-19-22", "2023-01-09-28", "2023-01-17-01", "2023-01-19-15", "2023-01-23-01", "2023-03-07-01", "2022-12-21-06", "2023-01-05-18", "2023-01-06-01", "2023-01-06-08", "2023-01-09-08", "2023-01-09-15", "2023-01-09-22", "2023-01-10-07", "2023-01-10-14", "2023-01-13-07", "2023-01-16-01", "2023-01-16-08", "2023-01-16-15", "2023-01-16-22", "2023-01-17-07", "2023-01-17-14", "2023-01-18-01"]
    datasets_prj_rim = ["2023-06-09-01", "2023-07-28-04", "2023-06-24-02", "2023-07-07-11", #"2023-08-07-01", # Don't use because worm stops moving about 5min in
                        "2023-06-24-11", "2023-07-07-18", "2023-08-18-11", "2023-06-24-28", "2023-07-11-02", "2023-08-22-08", "2023-07-12-01", "2023-07-01-09", "2023-07-13-01", "2023-06-09-10", "2023-07-07-01", "2023-08-07-16", "2023-08-22-01", "2023-08-23-23", "2023-08-25-02", "2023-09-15-01", "2023-09-15-08", "2023-08-18-18", "2023-08-19-01", "2023-08-23-09", "2023-08-25-09", "2023-09-01-01", "2023-08-31-03", "2023-07-01-01", "2023-07-01-23"]

    datasets_prj_aversion = [#"2023-03-30-01", 
        "2023-06-29-01", "2023-06-29-13", "2023-07-14-08", "2023-07-14-14", "2023-07-27-01", "2023-08-08-07", "2023-08-14-01", "2023-08-16-01", "2023-08-21-01", "2023-09-07-01", "2023-09-14-01", "2023-08-15-01", "2023-10-05-01", "2023-06-23-08", #"2023-12-11-01", 
                            "2023-06-21-01"]
    datasets_prj_5ht = ["2022-07-26-31", "2022-07-26-38", "2022-07-27-31", "2022-07-27-38", "2022-07-27-45", "2022-08-02-31", "2022-08-02-38", "2022-08-03-31"]
    datasets_prj_starvation = ["2023-05-25-08", "2023-05-26-08", "2023-06-05-10", "2023-06-05-17", "2023-07-24-27", "2023-09-27-14", "2023-05-25-01", "2023-05-26-01", "2023-07-24-12", "2023-07-24-20", "2023-09-12-01", "2023-09-19-01", "2023-09-29-19", "2023-10-09-01", "2023-09-13-02"]


    def get_folder(dataset):
        if dataset in datasets_prj_neuropal:
            return os.path.join(input_paths["prj_neuropal"], f"{dataset}_output")
        elif dataset in datasets_prj_rim:
            return os.path.join(input_paths["prj_rim"], f"{dataset}_output")
        elif dataset in datasets_prj_aversion:
            return os.path.join(input_paths["prj_aversion"], f"{dataset}_output")
        elif dataset in datasets_prj_5ht:
            return os.path.join(input_paths["prj_5ht"], f"{dataset}_output")
        elif dataset in datasets_prj_starvation:
            return os.path.join(input_paths["prj_starvation"], f"{dataset}_output")
        else:
            raise ValueError("Dataset not recognized")



    datasets_test = ['2023-08-22-01', '2023-07-07-18', '2023-07-01-23',  # RIM datasets
                 '2023-01-06-01', '2023-01-10-07', '2023-01-17-07', # Neuropal datasets
                 '2023-08-21-01', "2023-06-23-08", # Aversion datasets
                 '2022-07-27-38', # 5-HT datasets
                 '2023-10-09-01', '2023-09-13-02' # Starvation datasets
                 ]
    folders = [os.path.join(get_folder(x), "NRRD") for x in datasets_test]






    # assert os.path.exists(folder), "Folder does not exist"
    # assert not os.path.exists(os.path.join(folder, "liveTraces.h5")), "liveTraces.h5 already exists in the folder"

    predictor = _get_predictor(model, None, os.path.join(folder, "liveTraces_mergeshear_bgsubtract.h5"), config) # We can have just one predictor, because we are saving it all to one file

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
    
    #### TEMP
    # for folder, ds in zip(folders, datasets_test):
    #     assert os.path.exists(folder), "Folder does not exist"
    #     print(folder)
    #     predictor = _get_predictor(model, None, f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/traces_from_test_DSs/liveTraces_{ds}_mergeshear.h5", config) # We can have just one predictor, because we are saving it all to one file
    ####
    
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

                    
                    frame = int(file.split("t")[-1].split("_")[0])
                    # green = data

                elif file.endswith(".h5"):
                    # if "2022-07-27-31" not in file:
                    #     continue
                    
                    
                    with h5py.File(os.path.join(folder, file), 'r') as f:
                        red = np.ascontiguousarray(f["raw"][0])
                        green = red

                        # green = np.ascontiguousarray(f["raw_green"][0])
                        # labels = torch.tensor(f["label"][:], device=device)

                    print("Processing h5 " + file)
                    # t = int(file.split("_")[-1].split(".")[0])
                    frame = os.path.splitext(file)[0]
                    

                
                else:
                    continue
                
                D,H,W = red.shape
                print(f"Array size: D:{D} H:{H} W:{W}")

                # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/H5_raw_full_dataset/train/2022-07-27-31_{t}.h5", "r") as f:
                #     labels = torch.tensor(f["label"][:], device=device)

                # I'm just going to say from now on, [D, H, W] is the order we're going with
                chan_align_params = predictor.predict(red, green, chan_align_params=calc_chan_align_params, frame=frame)
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

    
