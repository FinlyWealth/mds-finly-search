### Install pgvector

In case your postgres is installed at `/Library/PostgreSQL/16`, not through home brew, try the following method to install pgvector.

In the finly conda environment, from any directory:

```{bash}
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

If you see any error like `make: arm64-apple-darwin20.0.0-clang: No such file or directory` when run the `make` command, try to run the following, and then run `make` again:

```{bash}
export PG_CONFIG=/Library/PostgreSQL/16/bin/pg_config
``` 

Now we need to copy the pgvector we installed in the finly conda enviornment into place where our Postgres database is installed.

```{bash}
# create the postgres extension folder
sudo mkdir -p /Library/PostgreSQL/16/share/extension

sudo cp /Users/{your_username}/miniforge3/envs/finly/share/extension/vector.control /Library/PostgreSQL/16/share/extension/

sudo cp /Users/{your_username}/miniforge3/envs/finly/lib/vector.dylib /Library/PostgreSQL/16/lib/postgresql/
```
