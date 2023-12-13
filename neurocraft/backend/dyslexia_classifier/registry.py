import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from neurocraft.params import *

def save_model(model: keras.Model = None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        #client = storage.Client()
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None

def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # # Get the latest model version name by the timestamp on disk
        # local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        # local_model_paths = glob.glob(f"{local_model_directory}/*")

        # if not local_model_paths:
        #     return None

        # most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        # print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        # latest_model = keras.models.load_model(most_recent_model_path_on_disk)


        # Get the directory of the current file (registry.py)
        current_dir = os.path.dirname(__file__)

        # Move up two directories to the root of your project
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Define the path to the .h5 file relative to the project root
        h5_file_path = os.path.join(project_root, 'model', '20231212-100653.h5')

        latest_model = keras.models.load_model(h5_file_path)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    else:
        return None
