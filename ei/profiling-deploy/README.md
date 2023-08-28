# Profiling and Deploy

## Profile your model
Reference [Edge Impulse Python SDK - profile your model](https://docs.edgeimpulse.com/docs/edge-impulse-python-sdk/01-python-sdk-with-tf-keras#profile-your-model)

To obtain list of devices define the [ei API key](https://raw.githubusercontent.com/edgeimpulse/notebooks/main/.assets/images/python-sdk-copy-ei-api-key.png) in `config.yaml'
```
ei:
  api_key: '<your-ei-key>'
```

and run

```
python list_profile_devices.py
```
To profile the model (default name: 'model.h5'), run:

```
python ei_profiling.py --wandb_run_path="<your-wandb-run-path>"  --device="<device>"
```
## Deploy your model
Reference [Edge Impulse Python SDK - deploy your model](https://docs.edgeimpulse.com/docs/edge-impulse-python-sdk/01-python-sdk-with-tf-keras#deploy-your-model)

To deploy the model to an Arduino target, run

```
python ei_create_lib.py --wandb_run_path="<your-wandb-run-path>"  --deploy_target="arduino"
```
 a zip file will be save in `./lib`