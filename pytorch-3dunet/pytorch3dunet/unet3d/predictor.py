import time

import h5py
import hdbscan
import numpy as np
import torch
from sklearn.cluster import MeanShift
import os

from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.datasets.shear_correction import shear_correction_torch, shear_correction_scikit
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.utils import remove_halo
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import functional as F
import euler_gpu

import cucim

logger = get_logger('UNet3DPredictor')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class LivePredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self, red_chan, green_chan = None, chan_align_params = None, frame = 0, labels = None):
        # def my_handler(prof): 
        #     print("Trace ready") 
        #     torch.profiler.tensorboard_trace_handler("/home/brian/data4/brian/freelyMoving/profiling")(prof) 
            
        # with torch.profiler.profile( 
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1), 
        #     on_trace_ready=my_handler, record_shapes=True, profile_memory=True, with_stack=True, 
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] ) as prof:
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        pred_threshold = self.config['model'].get('mask_threshold', 0.75) # Idk about this default val
        blur_size = self.config['model'].get('blur_size', 3)

        chan_batch_size = self.config['predictor'].get('chan_batch_size', 1)
        downsample_factor = self.config['predictor'].get('chan_downsample_factor', 1)

        # logger.info(f'Running prediction on {len(self.loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self._volume_shape(red_chan)
        prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
        self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
        logger.info(f'Using patch_halo: {patch_halo}')

        # create destination H5 file
        # h5_output_file = h5py.File(self.output_file, 'w')
        h5_output_file = h5py.File(self.output_file, 'a')
        
        # allocate prediction and normalization arrays
        # logger.info('Allocating prediction and normalization arrays...')
        # prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
        #                                                                     output_heads, h5_output_file)
        # compute_stream = torch.cuda.Stream(device=device)
        # green_stream   = torch.cuda.Stream(device=device)
        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.inference_mode():
            # send batch to device
            batch = torch.tensor(red_chan).to(device, non_blocking=True).to(torch.float32)
            batch = batch - torch.median(batch) # Background substract

            # batch = torch.permute(batch, (1, 2, 0))
            # batch = red_chan
            # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw_preds/preds_{frame}.h5", "w") as f:
            #     f.create_dataset("raw", data=batch.detach().cpu())

            # shift, error, phasediff = cucim.skimage.registration.phase_cross_correlation
            # shift, error, phasediff = skimage.registration.phase_cross_correlation
            # batch, shear_params_dict = shear_correction_torch(batch)

            # batch, shear_params_dict = shear_correction_scikit(batch.detach().cpu().numpy())
            # batch = torch.tensor(batch).to(device, non_blocking=True).to(torch.float32)

            # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw_preds/preds_{frame}.h5", "a") as f:
            #     f.create_dataset("raw_shearcorr", data=batch.detach().cpu())
            batch = batch.unsqueeze(0).unsqueeze(0)  # Add batch & chan dim

            # print(batch)
            # print(batch.shape)
            # forward pass
            assert batch.device == device, f"Batch device {batch.device} does not match model device {device}"
            predictions = self.model(batch)
            # assert predictions.shape[0] == 1, "Batch size other than 1 not supported in LivePredictor"

            # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw_preds-CrEnRe_noshear/preds_{frame}.h5", "w") as f:
            # with h5py.File(f"/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/predictions_mergeshear/{frame}.h5", "w") as f:
            #     f.create_dataset("predictions", data=predictions.detach().cpu())
            # return (0, 0, 0)

            num_classes = predictions.shape[1]
            mask = (predictions > pred_threshold).float()
            kernel = torch.ones((num_classes, 1, blur_size, blur_size, blur_size), device=device)
            kernel = kernel / kernel.numel()
            blurred_probs = F.conv3d(predictions, kernel, stride=1, padding=blur_size // 2, groups = num_classes) * mask
            blurred_probs = blurred_probs.squeeze(0)
            # print(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(probs, -1), -1), -1).expand(predictions.shape).shape)
            # Then max over layers
            max_prob, labels = blurred_probs.max(dim=0)
            # print(labels.shape)
            labels = labels.view(-1)  # Flatten to 1D

            ## Get trace from image. 
            # TODO: We don't have the green image accessible yet, so rewrite to handle it.
            if green_chan is None:
                print("Placholding Green Channel... no image given")
                green_chan = torch.ones_like(batch, device=device) # Placeholder 
            else:
                green_gpu = torch.tensor(green_chan).to(device, non_blocking = True).to(torch.float32)
                # green_gpu = green_gpu - 100
                green_gpu = green_gpu - torch.median(green_gpu) # Background substract

                # green_gpu = torch.permute(green_gpu, (1, 2, 0))

                #with torch.cuda.stream(green_stream):
                    #green_binned = F.avg_pool2d(green_gpu, 3, 3)

                # green_gpu, _ = shear_correction_torch(green_gpu, shear_params_dict)

                batch = batch[0,0]

                if not chan_align_params:
                    # TODO: Red-green alignment # TODO: Shouldn't matter if this is done before or after shear
                    ALIGN_XY_RANGE = np.linspace(-0.02, 0.02, 32, dtype=np.float32)
                    ALIGN_TH_RANGE = np.concatenate((np.linspace(0, .05, 8, dtype=np.float32), np.linspace(-0.05, 0, 8, dtype=np.float32)))
                    
                    # print(batch.shape)
                    # print(green_gpu.shape)
                    # DO we really need to MIP?
                    red_MIP = euler_gpu.max_intensity_projection_and_downsample_torch(batch, downsample_factor, projection_axis=0)
                    green_MIP = euler_gpu.max_intensity_projection_and_downsample_torch(green_gpu, downsample_factor, projection_axis=0)

                    # with h5py.File("/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw/chan_align_test.h5", "w") as f:
                    #     f.create_dataset("red_MIP", data=red_MIP.detach().cpu())
                    #     f.create_dataset("green_MIP", data=green_MIP.detach().cpu())

                    memory_dict = euler_gpu.initialize(red_MIP, green_MIP, ALIGN_XY_RANGE, ALIGN_XY_RANGE, ALIGN_TH_RANGE, chan_batch_size, device)
                    best_score, chan_align_params = euler_gpu.grid_search(memory_dict)

                ## Needing to make two memory dicts is really stupid. Also we're not going to align Z cause it shouldn't matter here
                ## TODO Get rid of this. Does O(n^3)
                memory_dict_transform = euler_gpu.initialize(
                    torch.empty_like(batch[0], dtype=torch.float32), # These need to be just one Z slice
                    torch.empty_like(batch[0], dtype=torch.float32), 
                    torch.empty(batch.shape[0], dtype=torch.float32),
                    torch.empty(batch.shape[0], dtype=torch.float32),
                    torch.empty(batch.shape[0], dtype=torch.float32),
                    batch.shape[0],
                    device
                )

                green_chan = euler_gpu.transform_image_3d_torch(green_gpu, memory_dict_transform, chan_align_params, device, 0)

                # with h5py.File("/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw/chan_align_test.h5", "a") as f:
                #         green_MIP = euler_gpu.max_intensity_projection_and_downsample_torch(green_chan, downsample_factor, projection_axis=2)
                #         f.create_dataset("green_MIP_align", data=green_MIP.detach().cpu())
                   

                # print(chan_align_params)

                ##### I'm thinking euler GPU rolling mean until it's under a certain threshold? Then apply, any maybe like check every couple frames? Depends on how fast it is? Could also do like a "warmup period of alignments"
                # green_chan = green_gpu

                ### TODO: Traces were slightly off 1 when running with same image. Check chan align or shear?

                
            # print(green_chan.shape)
            
            # Sum up predicted regions
            sums_sig = torch.bincount(labels, weights=green_chan.view(-1), minlength=out_channels)
            sums_mkr = torch.bincount(labels, weights=batch.view(-1), minlength=out_channels)
            counts   = torch.bincount(labels, minlength=out_channels)
            counts = counts.clamp_min(1) # Avoid div by 0 # Shouldn't affect results cause both should be 0 if count is 0

            signal = ((sums_sig / counts) / (sums_mkr / counts))

            # TODO: Account for laser intensity change
            
            # signal = ((sums_sig / counts) / (sums_mkr / counts).clamp_min(1)) # If we want to remove NaNs
            # prof.step()
        # folder = "/home/brian/data4/brian/freelyMoving/data/ACLL_unsheared/NRRD_raw/preds" 
        # with h5py.File(folder + str(len(os.listdir(folder))) + ".h5", 'a') as output_file:
        #     output_file.create_dataset("preds", data=labels.cpu().numpy().reshape(volume_shape))
        #     output_file.create_dataset("batch", data=batch.cpu().numpy().reshape(volume_shape))
            # output_file.create_dataset("class_probs", data=class_probs.cpu().numpy())
            # output_file.create_dataset("probs", data=probs.cpu().numpy())

        # h5_output_file.create_dataset("signal", data=signal.cpu().numpy())
        
        
        sig_DS = h5_output_file['signal'] if 'signal' in h5_output_file else h5_output_file.create_dataset("signal", (out_channels,0), maxshape=(out_channels,None), chunks=True)
        
        # Add chan for uncertain lab. Just for running on ground truth vals
        # sig_DS = h5_output_file['signal'] if 'signal' in h5_output_file else h5_output_file.create_dataset("signal", (out_channels + 1, 0), maxshape=(out_channels + 1,None), chunks=True)
        
        current_shape = sig_DS.shape#.cpu().numpy()
        new_length = current_shape[1] + 1
        sig_DS.resize(new_length, axis=1)
        # if signal.shape[0] == 186: # Chop off last "unsure" label channel. Just for running on ground truth vals

        sig_DS[:, -1] = signal.cpu().numpy()

        # close the output H5 file
        h5_output_file.close()
        return chan_align_params

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]

            logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            output_file.create_dataset(prediction_dataset, data=prediction_map)

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"



class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        def my_handler(prof): 
            print("Trace ready") 
            torch.profiler.tensorboard_trace_handler("/home/brian/data4/brian/freelyMoving/profiling")(prof) 
            
        with torch.profiler.profile( 
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1), 
            on_trace_ready=my_handler, record_shapes=True, profile_memory=True, with_stack=True, 
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] ) as prof:
            out_channels = self.config['model'].get('out_channels')
            if out_channels is None:
                out_channels = self.config['model']['dt_out_channels']

            prediction_channel = self.config.get('prediction_channel', None)
            if prediction_channel is not None:
                logger.info(f"Using only channel '{prediction_channel}' from the network output")

            device = self.config['device']
            output_heads = self.config['model'].get('output_heads', 1)

            pred_threshold = self.config['model'].get('mask_threshold', 0)

            logger.info(f'Running prediction on {len(self.loader)} batches...')

            # dimensionality of the the output predictions
            volume_shape = self._volume_shape(self.loader.dataset)
            prediction_maps_shape = (1,) + volume_shape

            logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

            patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
            self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
            logger.info(f'Using patch_halo: {patch_halo}')

            # create destination H5 file
            # h5_output_file = h5py.File(self.output_file, 'w')
            h5_output_file = h5py.File(self.output_file, 'a')
            
            # allocate prediction and normalization arrays
            # logger.info('Allocating prediction and normalization arrays...')
            # prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
            #                                                                     output_heads, h5_output_file)

            # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
            self.model.eval()
            # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
            self.model.testing = True
            # Run predictions on the entire input dataset
            with torch.no_grad():
                for batch, indices in self.loader:
                    # send batch to device
                    batch = batch.to(device)

                    # forward pass
                    predictions = self.model(batch)

                    #### PROCESS OUTPUT ON GPU ####
                    ## Get the predicted class for each voxel
                    max_prob, labels = predictions.max(dim=1)
                    
                    labels = torch.where(max_prob >= pred_threshold, labels, torch.zeros_like(labels))
                    # labels = labels.unsqueeze(1)  # Just so it has enough dims for other stuff. Should prob remove and remove those
                    labels = labels.view(-1)  # Flatten to 1D

                    ## Get trace from image. 
                    # TODO: We don't have the green image accessible yet, so rewrite to handle it.
                    green_img = torch.ones_like(batch, device=device) # Placeholder 

                    # TODO: Account for laser intensity change
                    
                    # Sum up predicted regions
                    sums_sig = torch.bincount(labels, weights=green_img.view(-1), minlength=out_channels)
                    sums_mkr = torch.bincount(labels, weights=batch.view(-1), minlength=out_channels)
                    counts   = torch.bincount(labels, minlength=out_channels)
                    counts = counts.clamp_min(1) # Avoid div by 0 # Shouldn't affect results cause both should be 0 if count is 0

                    signal = ((sums_sig / counts) / (sums_mkr / counts))
                    # signal = ((sums_sig / counts) / (sums_mkr / counts).clamp_min(1)) # If we want to remove NaNs



                    # # wrap predictions into a list if there is only one output head from the network
                    # if output_heads == 1:
                    #     predictions = [signal]

                    # # for each output head
                    # for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                    #                                                         normalization_masks):

                    #     # convert to numpy array
                    #     prediction = prediction.cpu().numpy()

                    #     # for each batch sample
                    #     for pred, index in zip(prediction, indices):
                    #         # save patch index: (C,D,H,W)
                    #         if prediction_channel is None:
                    #             channel_slice = slice(0, out_channels)
                    #         else:
                    #             channel_slice = slice(0, 1)
                    #         index = (channel_slice,) + tuple(index)

                    #         if prediction_channel is not None:
                    #             # use only the 'prediction_channel'
                    #             logger.info(f"Using channel '{prediction_channel}'...")
                    #             pred = np.expand_dims(pred[prediction_channel], axis=0)

                    #         logger.info(f'Saving predictions for slice:{index}...')

                    #         # remove halo in order to avoid block artifacts in the output probability maps
                    #         u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                    #         # accumulate probabilities into the output prediction array
                    #         prediction_map[u_index] += u_prediction
                    #         # count voxel visits for normalization
                    #         normalization_mask[u_index] += 1
                    #         prof.step()

        # save results to
        # self._save_results(prediction_maps, normalization_masks, output_heads, h5_output_file, self.loader.dataset)

        # h5_output_file.create_dataset("signal", data=signal.cpu().numpy())
        sig_DS = h5_output_file['signal'] if 'signal' in h5_output_file else h5_output_file.create_dataset("signal", (out_channels,0), maxshape=(out_channels,None), chunks=True)
        current_shape = sig_DS.shape#.cpu().numpy()
        new_length = current_shape[1] + 1
        sig_DS.resize(new_length, axis=1)
        sig_DS[:, -1] = signal.cpu().numpy()

        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]

            logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            output_file.create_dataset(prediction_dataset, data=prediction_map)

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"


class LazyPredictor(StandardPredictor):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this predictor
        is slower than the `StandardPredictor` it should only be used when the predicted volume does not fit into RAM.

        The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
        not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
        of the output head from the network.

        Args:
            model (Unet3D): trained 3D UNet model used for prediction
            data_loader (torch.utils.data.DataLoader): input data loader
            output_file (str): path to the output H5 file
            config (dict): global config dict
        """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # allocate datasets for probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        prediction_maps = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                       compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')
        normalization_masks = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='uint8', chunks=True,
                                       compression='gzip')
            for dataset_name in normalization_datasets]

        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        if dataset.mirror_padding:
            logger.warn(
                f'Mirror padding unsupported in LazyPredictor. Output predictions will be padded with pad_width: {dataset.pad_width}')

        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')

        # normalize the prediction_maps inside the H5
        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
                                                                                                 normalization_masks,
                                                                                                 prediction_datasets,
                                                                                                 normalization_datasets):
            # split the volume into 4 parts and load each into the memory separately
            logger.info(f'Normalizing {prediction_dataset}...')

            z, y, x = prediction_map.shape[1:]
            # take slices which are 1/27 of the original volume
            patch_shape = (z // 3, y // 3, x // 3)
            for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
                logger.info(f'Normalizing slice: {index}')
                prediction_map[index] /= normalization_mask[index]
                # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
                # when the patches overlap with each other
                normalization_mask[index] = 1

            logger.info(f'Deleting {normalization_dataset}...')
            del output_file[normalization_dataset]


class EmbeddingsPredictor(_AbstractPredictor):
    """
    Applies the embedding model on the given dataset and saves the result in the `output_file` in the H5 format.

    The resulting volume is the segmentation itself (not the embedding vectors) obtained by clustering embeddings
    with HDBSCAN or MeanShift algorithm patch by patch and then stitching the patches together.
    """

    def __init__(self, model, loader, output_file, config, clustering, iou_threshold=0.7, noise_label=-1, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

        self.iou_threshold = iou_threshold
        self.noise_label = noise_label
        self.clustering = clustering

        assert clustering in ['hdbscan', 'meanshift'], 'Only HDBSCAN and MeanShift are supported'
        logger.info(f'IoU threshold: {iou_threshold}')

        self.clustering_name = clustering
        self.clustering = self._get_clustering(clustering, kwargs)

    def predict(self):
        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(self.loader)} patches...')

        # dimensionality of the the output segmentation
        volume_shape = self._volume_shape(self.loader.dataset)

        logger.info(f'The shape of the output segmentation (DHW): {volume_shape}')

        logger.info('Allocating segmentation array...')
        # initialize the output prediction arrays
        output_segmentations = [np.zeros(volume_shape, dtype='int32') for _ in range(output_heads)]
        # initialize visited_voxels arrays
        visited_voxels_arrays = [np.zeros(volume_shape, dtype='uint8') for _ in range(output_heads)]

        # Sets the module in evaluation mode explicitly
        self.model.eval()
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # logger.info(f'Predicting embeddings for slice:{index}')

                # send batch to device
                batch = batch.to(device)
                # forward pass
                embeddings = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    embeddings = [embeddings]

                for prediction, output_segmentation, visited_voxels_array in zip(embeddings, output_segmentations,
                                                                                 visited_voxels_arrays):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # iterate sequentially because of the current simple stitching that we're using
                    for pred, index in zip(prediction, indices):
                        # convert embeddings to segmentation with hdbscan clustering
                        segmentation = self._embeddings_to_segmentation(pred)
                        # stitch patches
                        self._merge_segmentation(segmentation, index, output_segmentation, visited_voxels_array)

        # save results
        with h5py.File(self.output_file, 'w') as output_file:
            prediction_datasets = self._get_output_dataset_names(output_heads,
                                                                 prefix=f'segmentation/{self.clustering_name}')
            for output_segmentation, prediction_dataset in zip(output_segmentations, prediction_datasets):
                logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
                output_file.create_dataset(prediction_dataset, data=output_segmentation, compression="gzip")

    def _embeddings_to_segmentation(self, embeddings):
        """
        Cluster embeddings vectors with HDBSCAN and return the segmented volume.

        Args:
            embeddings (ndarray): 4D (CDHW) embeddings tensor
        Returns:
            3D (DHW) segmentation
        """
        # shape of the output segmentation
        output_shape = embeddings.shape[1:]
        # reshape (C, D, H, W) -> (C, D * H * W) and transpose -> (D * H * W, C)
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

        logger.info('Clustering embeddings...')
        # perform clustering and reshape in order to get the segmentation volume
        start = time.time()
        clusters = self.clustering.fit_predict(flattened_embeddings).reshape(output_shape)
        logger.info(
            f'Number of clusters found by {self.clustering}: {np.max(clusters)}. Duration: {time.time() - start} sec.')
        return clusters

    def _merge_segmentation(self, segmentation, index, output_segmentation, visited_voxels_array):
        """
        Given the `segmentation` patch, its `index` in the `output_segmentation` array and the array visited voxels
        merge the segmented patch (`segmentation`) into the `output_segmentation`

        Args:
            segmentation (ndarray): segmented patch
            index (tuple): position of the patch inside `output_segmentation` volume
            output_segmentation (ndarray): current state of the output segmentation
            visited_voxels_array (ndarray): array of voxels visited so far (same size as `output_segmentation`); visited
                voxels will be marked by a number greater than 0
        """
        index = tuple(index)
        # get new unassigned label
        max_label = np.max(output_segmentation) + 1
        # make sure there are no clashes between current segmentation patch and the output_segmentation
        # but keep the noise label
        noise_mask = segmentation == self.noise_label
        segmentation += int(max_label)
        segmentation[noise_mask] = self.noise_label
        # get the overlap mask in the current patch
        overlap_mask = visited_voxels_array[index] > 0
        # get the new labels inside the overlap_mask
        new_labels = np.unique(segmentation[overlap_mask])
        merged_labels = self._merge_labels(output_segmentation[index], new_labels, segmentation)
        # relabel new segmentation with the merged labels
        for current_label, new_label in merged_labels:
            segmentation[segmentation == new_label] = current_label
        # update the output_segmentation
        output_segmentation[index] = segmentation
        # visit the patch
        visited_voxels_array[index] += 1

    def _merge_labels(self, current_segmentation, new_labels, new_segmentation):
        def _most_frequent_label(labels):
            unique, counts = np.unique(labels, return_counts=True)
            ind = np.argmax(counts)
            return unique[ind]

        result = []
        # iterate over new_labels and merge regions if the IoU exceeds a given threshold
        for new_label in new_labels:
            # skip 'noise' label assigned by hdbscan
            if new_label == self.noise_label:
                continue
            new_label_mask = new_segmentation == new_label
            # get only the most frequent overlapping label
            most_frequent_label = _most_frequent_label(current_segmentation[new_label_mask])
            # skip 'noise' label
            if most_frequent_label == self.noise_label:
                continue
            current_label_mask = current_segmentation == most_frequent_label
            # compute Jaccard index
            iou = np.bitwise_and(new_label_mask, current_label_mask).sum() / np.bitwise_or(new_label_mask,
                                                                                           current_label_mask).sum()
            if iou > self.iou_threshold:
                # merge labels
                result.append((most_frequent_label, new_label))

        return result

    def _get_clustering(self, clustering_alg, kwargs):
        logger.info(f'Using {clustering_alg} for clustering')

        if clustering_alg == 'hdbscan':
            min_cluster_size = kwargs.get('min_cluster_size', 50)
            min_samples = kwargs.get('min_samples', None),
            metric = kwargs.get('metric', 'euclidean')
            cluster_selection_method = kwargs.get('cluster_selection_method', 'eom')

            logger.info(f'HDBSCAN params: min_cluster_size: {min_cluster_size}, min_samples: {min_samples}')
            return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                                   cluster_selection_method=cluster_selection_method)
        else:
            bandwidth = kwargs['bandwidth']
            logger.info(f'MeanShift params: bandwidth: {bandwidth}, bin_seeding: True')
            # use fast MeanShift with bin seeding
            return MeanShift(bandwidth=bandwidth, bin_seeding=True)