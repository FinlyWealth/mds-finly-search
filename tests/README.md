# Run tests

- Install the required packages

    ```{bash}
    pip install pytest-cov testcontainers[postgresql] pytest psycopg2-binary
    ```

- Run tests from the root directory

    ```{bash}
    pytest
    ```

- Check the coverage for a specific directory

    ```{bash}
    pytest --cov=preprocess tests/
    ```
