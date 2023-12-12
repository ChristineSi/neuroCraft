from google.cloud import storage
from neurocraft.params import *
#import time

client = storage.Client(project=GCP_PROJECT)
bucket = client.bucket(BUCKET_NAME)
#timestamp = time.strftime("%Y%m%d-%H%M%S")
#model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
#model_filename = model_path.split("/")[-1]
blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
latest_blob = max(blobs, key=lambda x: x.updated)
latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
latest_blob.download_to_filename(latest_model_path_to_save)

#print(latest_blob)
#print(model_filename)
