This container is a simple spark application that reads from a postgres database and processes the data.

Currently there is only one operation, a simple spam filter that gets added to the dataframe.

To run this container, first build it with the following command:

```
docker build -t sparkapp .
```

Then, run the container with the following command (replacing the env vars with your own):

```
docker run --name spam_filter -e DB_USER="chris" -e DB_PWD="password" -e DB_HOST="csci6010postgres.mjpmorse.dev" -e DB_PORT="5433" -e DB_DATABASE="postgres" sparkapp
```