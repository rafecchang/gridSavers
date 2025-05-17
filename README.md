# Grid Savers
A notebook that modeled V2G battery arbtitrage at CAISO. 

### Look around 

View at: https://rafecchang.github.io/gridSavers/

### Look deeper 

Some quick steps to run the notebook locally. 

1. Clone the repository locally. In your terminal run:

    ```console
    $ git clone https://github.com/rafecchang/gridSavers.git 
    ```

2. Create and activate the `conda` environment. In the root of the repository run:
    ```console
    $ conda env create --file environment.yml
    ```

    ```console
    $ conda activate gridSavers
    ```

3. Create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-fix-or-feature
    ```

4. To preprocess data, from the root of the directory (`gridSavers`) run: 

    ```console
    $ python src/preprocess.py
    ```

5. Open the `notebooks/gridSavers.ipynb` file to make changes: 

    ```console
    $ code notebooks/gridSavers.ipynb
    ```
