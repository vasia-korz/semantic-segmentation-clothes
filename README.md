# People Clothing Segmentation

The dataset as well as the descriptiont of the problem can be found on [Kaggle](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation).

## DVC

The project is configured to work with `OAuth2`.

In order to be able to authenticate in the existing project, you have to create `.dvc/config.local` file with the following layout:

```dvc
['remote "origin"']
    gdrive_client_id = <YOUR-CLIENT-ID>
    gdrive_client_secret = <YOUR-CLIENT-SECRET>
```

However, if you are willing use the project from scratch (without access to the remote data storage), you would have to go through the [guide](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) to set up your own one.
