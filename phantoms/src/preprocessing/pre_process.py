from Dinor_revision.git_tbmf.phantoms.src.utils.utils import *
import os
import numpy as np
import glob


config = {
    "input_path" : "/home/sahar/Models/Dinor_revision/personal_git/phantoms", # Change if needed
    "input_name" : "input.mat",
    "phantom_name" : "scan12",
    "labels_name" : "labels.mat",
    "need_output_file" : False,
    # Black removal
    "min_keep_slices" : 5,
    "threshold" : 30,
    "axis" : 1, # axial

    "output_folder" : "/home/sahar/Models/Dinor_revision/personal_git/phantoms/phantoms_clean", # Change if needed
    # Segment phantom circle
    "sam_checkpoint": "/home/sahar/Models/Dinor_revision/personal_git/phantoms/sam_checkpoints/sam_vit_h_4b8939.pth",
    "wanted_slices" : [[slice(41, 46), slice(62, 69)]], # Valid only for phantom 12
    # 'wanted_slices': [[slice(47, 52), slice(70, 77)]],
    "area_threshold_low": 2250,     # minimum and maximum areas of the phantom mask to be found
    "area_threshold_high" : 3000,
    # Vial parameters
    "vial_area_threshold_low": 22,
    "vial_area_threshold_high": 79,
    "default_num_vials": 6,
    "tube_radius": 5,

    # Cropping parameters
    "target_height": 80,  # Target H dimension
    "target_width": 80,  # Target W dimension

    # Skip processing flags
    "skip_segmentation": True,  # Set to True to load existing segmentation checkpoints
    "skip_cropping": True,  # Set to True to load existing cropping checkpoints
    "skip_labels_processing": True,  # Set to True to load existing labels checkpoints

    # Paths for loading existing checkpoints (only needed if skip flags are True)
    "existing_results_path": "/home/sahar/Models/Dinor_revision/personal_git/phantoms/phantoms_clean",
    # Path where existing checkpoints are stored

    # fit data to model:
    "needs_preparation_for_tbmf": True, # Change to false if you only want segmentation
    "parameter_map" : np.array(
                [[2, 2, 1.7, 1.5, 1.2, 1.2, 3, 0.5, 3, 1, 2.2, 3.2, 1.5, 0.7, 1.5, 2.2, 2.5, 1.2, 3, 0.2,
                  1.5, 2.5, 0.7, 4, 3.2, 3.5, 1.5, 2.7, 0.7, 0.5],
                 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]),
    "need_augmentation" : False,
    "views" : ["axial"], # Add sagittal and coronal if needed

    "model_type" : "model2", # Options : model1, model2
    "segmented": False,
    "window_size": 6,
    "prediction_offset": 6,
    "embedding_window": 12,

    # Model 2 labels creations:
    "generate_segmented_labels": True,  # Set to False if you don't want to generate them

    "default_parameter_values": {
        "ph_values": [5.5, 6, 5.0, 4.0, 5.0, 5.0],
        "mM_values": [50, 50, 50, 50, 25, 100],
        "T1_values": [2865, 3160, 2950, 3087, 3308, 2890],
        "T2_values": [663, 856, 639, 559, 478, 999],
        "ksw_values": [441, 437, 670, 384, 1066, 204],
        "fs_values": [50, 25, 50, 100, 50, 50]
    },
    "model2_type": {1 : ['pH', 'mM'],
                    2 : ['T1', 'T2'],
                    3: ['ksw', 'fs'],
                    4: ['B0', 'B1']
                    },
    "B_maps_path": "/home/sahar/Models/Dinor_revision/personal_git/phantoms/scan12/B_maps.h5"
}

# Initialize both segmenter and labels processor
segmenter = PhantomSegmenter(config)
labels_processor = LabelsProcessor(config)
cropper = PhantomCropper(config)
model1_fitting = Model1_Dataset(config)
segmented_labels_generator = SegmentedLabelsGenerator(config)
model2_fitting = Model2_Dataset(config)


# Find input and label files
if config["phantom_name"] is None:
    input_paths = glob.glob(os.path.join(config["input_path"], "*/input*.mat"))
else:
    input_paths = glob.glob(os.path.join(config["input_path"], config["phantom_name"], "input*.mat"))

for input_path in input_paths:
    if config["phantom_name"] is None:
        config["phantom_name"] = input_path.split("/")[-2]
    # Find corresponding label file
    input_dir = os.path.dirname(input_path)
    label_path = os.path.join(input_dir, config["labels_name"])

    if not os.path.exists(label_path):
        print(f"Warning: No label file found at {label_path}, processing input only...")
        label_path = None

    print(f"\nProcessing:")
    print(f"  Input: {input_path}")
    if label_path:
        print(f"  Labels: {label_path}")

    # Check if we should skip segmentation and load existing checkpoints
    if config.get("skip_segmentation", False):
        print("\n" + "="*60)
        print("SKIPPING SEGMENTATION - LOADING EXISTING RESULTS")
        print("="*60)
        phantom_results = load_existing_phantom_results(config)
    else:
        # Process input data (your existing code)
        data = load_mat_volume(input_path)
        data = np.transpose(data, (0, 2, 1, 3))
        data = remove_black_slices(data, axis=config["axis"], threshold=config["threshold"],
                                   min_keep_percentage=config["min_keep_slices"])
        print(f"Input data shape after processing: {data.shape}")
        phantom_results = segmenter.process_phantom_complete(data)

    # Check if we should skip labels processing and load existing checkpoints
    if config.get("skip_labels_processing", False) and label_path:
        print("\n" + "="*60)
        print("SKIPPING LABELS PROCESSING - LOADING EXISTING RESULTS")
        print("="*60)
        labels_results = load_existing_labels_results(config)
    else:
        # Process labels if available
        labels_results = None
        if label_path:
            labels_results = labels_processor.process_complete_pipeline(
                label_path,
                phantom_results,
                wanted_slices=config.get("wanted_slices")
            )

    # Check if we should skip cropping and load existing checkpoints
    if config.get("skip_cropping", False):
        print("\n" + "="*60)
        print("SKIPPING CROPPING - LOADING EXISTING RESULTS")
        print("="*60)
        cropped_phantom_results, cropped_labels_results = load_existing_cropped_results(config)
    else:
        # ADD THIS: Crop all checkpoints
        cropped_phantom_results, cropped_labels_results = cropper.process_complete_cropping(
            phantom_results, labels_results
        )

    # Results now contain:
    # - phantom_results: Original segmentation checkpoints
    # - labels_results: Original labels checkpoints (if available)
    # - cropped_phantom_results: Cropped phantom data, masks, and vial images
    # - cropped_labels_results: Cropped labels data (if available)

    print(f"\nCompleted processing for {config.get('phantom_name', 'phantom')}")
    print("=" * 60)

    if config['needs_preparation_for_tbmf']:
        # Fit data into the model
        if config["model_type"] == "model1":
            if config["segmented"]:
                model1_fitting.process_data_for_model_1(
                    phantom_4D=cropped_phantom_results["vial_masked_images_4d"])
            else:
                # Always use phantom-masked data to match training data
                phantom_masked_data = cropped_phantom_results["clean_phantom"] * \
                                      cropped_phantom_results["phantom_masks_3d"][np.newaxis, :, :, :]
                model1_fitting.process_data_for_model_1(phantom_4D=phantom_masked_data)
                # In your main processing loop, modify the Model 2 section:
        elif config["model_type"] == "model2":
            # Model 2 preparation - Generate segmented labels
            try:
                print("PREPARING DATA FOR MODEL2 - GENERATING SEGMENTED LABELS")

                # Check if segmenter has the required SAM components
                if not hasattr(segmenter, 'sam') or segmenter.sam is None:
                    print("Warning: SAM model not initialized. Initializing now...")
                    segmenter.initialize_sam_model()

                # Generate segmented parameter maps for model2
                labels_maps = segmented_labels_generator.generate_segmented_labels(
                    cropped_phantom_results,  # Use cropped checkpoints for model2
                    segmenter.sam,
                    segmenter.mask_generator
                )

                if labels_maps is not None:
                    while True:
                        try:
                            labels = input(f"Enter model type: 1 for pH-mM, 2 for T1,T2, 3 for fs-ksw 4 for B0,B1:")
                            labels = int(labels)
                            if labels not in [1, 2, 3, 4]:
                                print(
                                    f"Error: Expected to choose 1,2,3 or 4 choose again")
                                continue
                            break
                        except ValueError:
                            print("Error: Please enter valid numeric values")


                    # Prepare data for Model 2 with chosen type
                    if labels != 4:
                        phantom_masked_data = cropped_phantom_results["vial_masked_images_4d"]
                        labels_data = np.zeros((2, phantom_masked_data.shape[1], phantom_masked_data.shape[2],
                                                phantom_masked_data.shape[3]))
                        labels_data[0] = labels_maps[config["model2_type"][labels][0]]
                        labels_data[1] = labels_maps[config["model2_type"][labels][1]]


                    else:
                        phantom_masked_data = cropped_phantom_results["clean_phantom"] * \
                                              cropped_phantom_results["phantom_masks_3d"][np.newaxis, :, :, :]

                        B_data = h5py.File(config["B_maps_path"])["res"][:]
                        labels_data = np.transpose(B_data, (0,3,1,2))


                    folder_name = str(config["model2_type"][labels][0]) +"_"+ str(config["model2_type"][labels][1])
                    model2_fitting.process_data_for_model_2(
                        phantom_4D=phantom_masked_data,
                        labels_maps=labels_data, folder_name=folder_name)

                    print("Model2 data preparation completed successfully!")

                else:
                    print("Failed to generate parameter maps. Model2 preparation aborted.")

            except KeyboardInterrupt:
                print("\nModel2 preparation interrupted by user.")
            except Exception as e:
                print(f"Error during Model2 preparation: {str(e)}")

        else:
            print(f"Unknown model_type: {config['model_type']}. Supported types: 'model1', 'model2'")



