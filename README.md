# MATERIALS RECOGNITION

## ENV SETUP

rename the **.env.example** file at the project's root to **.env** and fill it with your values.

*your can search the variable directly in the code to know where it's used*

## BUILD THE DATASETS

```
python3 dataset_builder.py <1|2|3>
```

* **1** to create a **PVC vs all** dataset
* **2** to create a **PVC vs wood vs glass vs joint vs other** dataset
* **3** to create a **PVC vs wood vs glass vs joint vs PE/PA/PS vs other** dataset

For each dataset, there will be a **train** and a **test** *.bin* file created.

## TESTS

to run the tests:
```
python3 -m unittest discover
```